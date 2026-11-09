//===- ScheduleDAG.cpp - Implement the ScheduleDAG class ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Implements the ScheduleDAG class, which is a base class used by
/// scheduling implementation classes.
//
//===----------------------------------------------------------------------===//

#include "llvm/CodeGen/ScheduleDAG.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/Statistic.h"
#include "llvm/CodeGen/MachineFunction.h"
#include "llvm/CodeGen/ScheduleHazardRecognizer.h"
#include "llvm/CodeGen/SelectionDAGNodes.h"
#include "llvm/CodeGen/TargetInstrInfo.h"
#include "llvm/CodeGen/TargetRegisterInfo.h"
#include "llvm/CodeGen/TargetSubtargetInfo.h"
#include "llvm/Config/llvm-config.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/Compiler.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cassert>
#include <functional>
#include <iterator>
#include <limits>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "pre-RA-sched"

STATISTIC(NumNewPredsAdded, "Number of times a  single predecessor was added");
STATISTIC(NumTopoInits,
          "Number of times the topological order has been recomputed");
STATISTIC(NumReachQueries, "Number of reachability queries");
STATISTIC(NumReachSearches,
          "Number of reachability queries that needed a search");

#ifndef NDEBUG
static cl::opt<bool> StressSchedOpt(
  "stress-sched", cl::Hidden, cl::init(false),
  cl::desc("Stress test instruction scheduling"));
#endif

static cl::opt<unsigned> TopoSupportiveVertices(
    "sched-topo-supportive-vertices", cl::Hidden, cl::init(4),
    cl::desc("Number of supportive vertices used to answer reachability "
             "queries on scheduling DAGs (at most 32)"));

/// Recomputing from scratch is likely more efficient than applying too many
/// queued updates one-by-one. The cut-off is arbitrarily chosen.
static constexpr unsigned MaxQueuedUpdates = 10;

/// Number of cached reachability rows kept for each direction.
static constexpr unsigned NumReachRows = 16;

void SchedulingPriorityQueue::anchor() {}

ScheduleDAG::ScheduleDAG(MachineFunction &mf)
    : TM(mf.getTarget()), TII(mf.getSubtarget().getInstrInfo()),
      TRI(mf.getSubtarget().getRegisterInfo()), MF(mf),
      MRI(mf.getRegInfo()) {
#ifndef NDEBUG
  StressSched = StressSchedOpt;
#endif
}

ScheduleDAG::~ScheduleDAG() = default;

void ScheduleDAG::clearDAG() {
  SUnits.clear();
  EntrySU = SUnit();
  ExitSU = SUnit();
}

const MCInstrDesc *ScheduleDAG::getNodeDesc(const SDNode *Node) const {
  if (!Node || !Node->isMachineOpcode()) return nullptr;
  return &TII->get(Node->getMachineOpcode());
}

LLVM_DUMP_METHOD void SDep::dump(const TargetRegisterInfo *TRI) const {
  switch (getKind()) {
  case Data:   dbgs() << "Data"; break;
  case Anti:   dbgs() << "Anti"; break;
  case Output: dbgs() << "Out "; break;
  case Order:  dbgs() << "Ord "; break;
  }

  switch (getKind()) {
  case Data:
    dbgs() << " Latency=" << getLatency();
    if (TRI && isAssignedRegDep())
      dbgs() << " Reg=" << printReg(getReg(), TRI);
    break;
  case Anti:
  case Output:
    dbgs() << " Latency=" << getLatency();
    break;
  case Order:
    dbgs() << " Latency=" << getLatency();
    switch(Contents.OrdKind) {
    case Barrier:      dbgs() << " Barrier"; break;
    case MayAliasMem:
    case MustAliasMem: dbgs() << " Memory"; break;
    case Artificial:   dbgs() << " Artificial"; break;
    case Weak:         dbgs() << " Weak"; break;
    case Cluster:      dbgs() << " Cluster"; break;
    }
    break;
  }
}

bool SUnit::addPred(const SDep &D, bool Required) {
  // If this node already has this dependence, don't add a redundant one.
  for (SDep &PredDep : Preds) {
    // Zero-latency weak edges may be added purely for heuristic ordering. Don't
    // add them if another kind of edge already exists.
    if (!Required && PredDep.getSUnit() == D.getSUnit())
      return false;
    if (PredDep.overlaps(D)) {
      // Extend the latency if needed. Equivalent to
      // removePred(PredDep) + addPred(D).
      if (PredDep.getLatency() < D.getLatency()) {
        SUnit *PredSU = PredDep.getSUnit();
        // Find the corresponding successor in N.
        SDep ForwardD = PredDep;
        ForwardD.setSUnit(this);
        for (SDep &SuccDep : PredSU->Succs) {
          if (SuccDep == ForwardD) {
            SuccDep.setLatency(D.getLatency());
            break;
          }
        }
        PredDep.setLatency(D.getLatency());
        // Changing latency, dirty the involved SUnits.
        this->setDepthDirty();
        D.getSUnit()->setHeightDirty();
      }
      return false;
    }
  }
  // Now add a corresponding succ to N.
  SDep P = D;
  P.setSUnit(this);
  SUnit *N = D.getSUnit();
  // Update the bookkeeping.
  if (D.getKind() == SDep::Data) {
    assert(NumPreds < std::numeric_limits<unsigned>::max() &&
           "NumPreds will overflow!");
    assert(N->NumSuccs < std::numeric_limits<unsigned>::max() &&
           "NumSuccs will overflow!");
    ++NumPreds;
    ++N->NumSuccs;
  }
  if (!N->isScheduled) {
    if (D.isWeak()) {
      ++WeakPredsLeft;
    }
    else {
      assert(NumPredsLeft < std::numeric_limits<unsigned>::max() &&
             "NumPredsLeft will overflow!");
      ++NumPredsLeft;
    }
  }
  if (!isScheduled) {
    if (D.isWeak()) {
      ++N->WeakSuccsLeft;
    }
    else {
      assert(N->NumSuccsLeft < std::numeric_limits<unsigned>::max() &&
             "NumSuccsLeft will overflow!");
      ++N->NumSuccsLeft;
    }
  }
  Preds.push_back(D);
  N->Succs.push_back(P);
  this->setDepthDirty();
  N->setHeightDirty();
  return true;
}

void SUnit::removePred(const SDep &D) {
  // Find the matching predecessor.
  SmallVectorImpl<SDep>::iterator I = llvm::find(Preds, D);
  if (I == Preds.end())
    return;
  // Find the corresponding successor in N.
  SDep P = D;
  P.setSUnit(this);
  SUnit *N = D.getSUnit();
  SmallVectorImpl<SDep>::iterator Succ = llvm::find(N->Succs, P);
  assert(Succ != N->Succs.end() && "Mismatching preds / succs lists!");
  // Update the bookkeeping.
  if (P.getKind() == SDep::Data) {
    assert(NumPreds > 0 && "NumPreds will underflow!");
    assert(N->NumSuccs > 0 && "NumSuccs will underflow!");
    --NumPreds;
    --N->NumSuccs;
  }
  if (!N->isScheduled) {
    if (D.isWeak()) {
      assert(WeakPredsLeft > 0 && "WeakPredsLeft will underflow!");
      --WeakPredsLeft;
    } else {
      assert(NumPredsLeft > 0 && "NumPredsLeft will underflow!");
      --NumPredsLeft;
    }
  }
  if (!isScheduled) {
    if (D.isWeak()) {
      assert(N->WeakSuccsLeft > 0 && "WeakSuccsLeft will underflow!");
      --N->WeakSuccsLeft;
    } else {
      assert(N->NumSuccsLeft > 0 && "NumSuccsLeft will underflow!");
      --N->NumSuccsLeft;
    }
  }
  N->Succs.erase(Succ);
  Preds.erase(I);
  this->setDepthDirty();
  N->setHeightDirty();
}

void SUnit::setDepthDirty() {
  if (!isDepthCurrent) return;
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *SU = WorkList.pop_back_val();
    SU->isDepthCurrent = false;
    for (SDep &SuccDep : SU->Succs) {
      SUnit *SuccSU = SuccDep.getSUnit();
      if (SuccSU->isDepthCurrent)
        WorkList.push_back(SuccSU);
    }
  } while (!WorkList.empty());
}

void SUnit::setHeightDirty() {
  if (!isHeightCurrent) return;
  SmallVector<SUnit*, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *SU = WorkList.pop_back_val();
    SU->isHeightCurrent = false;
    for (SDep &PredDep : SU->Preds) {
      SUnit *PredSU = PredDep.getSUnit();
      if (PredSU->isHeightCurrent)
        WorkList.push_back(PredSU);
    }
  } while (!WorkList.empty());
}

void SUnit::setDepthToAtLeast(unsigned NewDepth) {
  if (NewDepth <= getDepth())
    return;
  setDepthDirty();
  Depth = NewDepth;
  isDepthCurrent = true;
}

void SUnit::setHeightToAtLeast(unsigned NewHeight) {
  if (NewHeight <= getHeight())
    return;
  setHeightDirty();
  Height = NewHeight;
  isHeightCurrent = true;
}

/// Calculates the maximal path from the node to the entry.
void SUnit::ComputeDepth() {
  // Iterative post-order DFS along Preds. Pushing one pred at a time and
  // finalizing on pop. A node on the stack cannot reappear as a pred of any
  // descendant.
  SmallVector<SUnit *, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *Cur = WorkList.back();
    bool Descended = false;
    for (const SDep &PredDep : Cur->Preds) {
      SUnit *PredSU = PredDep.getSUnit();
      if (!PredSU->isDepthCurrent) {
        WorkList.push_back(PredSU);
        Descended = true;
        break;
      }
    }
    if (Descended)
      continue;
    WorkList.pop_back();
    unsigned MaxPredDepth = 0;
    for (const SDep &PredDep : Cur->Preds)
      MaxPredDepth = std::max(MaxPredDepth,
                              PredDep.getSUnit()->Depth + PredDep.getLatency());
    Cur->Depth = MaxPredDepth;
    Cur->isDepthCurrent = true;
  } while (!WorkList.empty());
}

/// Calculates the maximal path from the node to the exit.
void SUnit::ComputeHeight() {
  // See ComputeDepth; this is the mirror image walking Succs.
  SmallVector<SUnit *, 8> WorkList;
  WorkList.push_back(this);
  do {
    SUnit *Cur = WorkList.back();
    bool Descended = false;
    for (const SDep &SuccDep : Cur->Succs) {
      SUnit *SuccSU = SuccDep.getSUnit();
      if (!SuccSU->isHeightCurrent) {
        WorkList.push_back(SuccSU);
        Descended = true;
        break;
      }
    }
    if (Descended)
      continue;
    WorkList.pop_back();
    unsigned MaxSuccHeight = 0;
    for (const SDep &SuccDep : Cur->Succs)
      MaxSuccHeight = std::max(MaxSuccHeight, SuccDep.getSUnit()->Height +
                                                  SuccDep.getLatency());
    Cur->Height = MaxSuccHeight;
    Cur->isHeightCurrent = true;
  } while (!WorkList.empty());
}

void SUnit::biasCriticalPath() {
  if (NumPreds < 2)
    return;

  SUnit::pred_iterator BestI = Preds.begin();
  unsigned MaxDepth = BestI->getSUnit()->getDepth();
  for (SUnit::pred_iterator I = std::next(BestI), E = Preds.end(); I != E;
       ++I) {
    if (I->getKind() == SDep::Data && I->getSUnit()->getDepth() > MaxDepth) {
      MaxDepth = I->getSUnit()->getDepth();
      BestI = I;
    }
  }
  if (BestI != Preds.begin())
    std::swap(*Preds.begin(), *BestI);
}

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
raw_ostream &llvm::operator<<(raw_ostream &OS, const SUnit &SU) {
  assert(!SU.isBoundaryNode() &&
         "use ScheduleDAG::dumpNodeName for boundary nodes");
  return OS << "SU(" << SU.NodeNum << ")";
}
#endif

#if !defined(NDEBUG) || defined(LLVM_ENABLE_DUMP)
LLVM_DUMP_METHOD void SUnit::dumpAttributes() const {
  dbgs() << "  # preds left       : " << NumPredsLeft << "\n";
  dbgs() << "  # succs left       : " << NumSuccsLeft << "\n";
  if (WeakPredsLeft)
    dbgs() << "  # weak preds left  : " << WeakPredsLeft << "\n";
  if (WeakSuccsLeft)
    dbgs() << "  # weak succs left  : " << WeakSuccsLeft << "\n";
  dbgs() << "  # rdefs left       : " << NumRegDefsLeft << "\n";
  dbgs() << "  Latency            : " << Latency << "\n";
  dbgs() << "  Depth              : " << getDepth() << "\n";
  dbgs() << "  Height             : " << getHeight() << "\n";
}

LLVM_DUMP_METHOD void ScheduleDAG::dumpNodeName(const SUnit &SU) const {
  if (&SU == &EntrySU)
    dbgs() << "EntrySU";
  else if (&SU == &ExitSU)
    dbgs() << "ExitSU";
  else
    dbgs() << SU;
}

LLVM_DUMP_METHOD void ScheduleDAG::dumpNodeAll(const SUnit &SU) const {
  dumpNode(SU);
  SU.dumpAttributes();
  if (SU.isClustered())
    dbgs() << "  Parent Cluster Index: " << SU.ParentClusterIdx << '\n';

  if (SU.Preds.size() > 0) {
    dbgs() << "  Predecessors:\n";
    for (const SDep &Dep : SU.Preds) {
      dbgs() << "    ";
      dumpNodeName(*Dep.getSUnit());
      dbgs() << ": ";
      Dep.dump(TRI);
      dbgs() << '\n';
    }
  }
  if (SU.Succs.size() > 0) {
    dbgs() << "  Successors:\n";
    for (const SDep &Dep : SU.Succs) {
      dbgs() << "    ";
      dumpNodeName(*Dep.getSUnit());
      dbgs() << ": ";
      Dep.dump(TRI);
      dbgs() << '\n';
    }
  }
}
#endif

#ifndef NDEBUG
unsigned ScheduleDAG::VerifyScheduledDAG(bool isBottomUp) {
  bool AnyNotSched = false;
  unsigned DeadNodes = 0;
  for (const SUnit &SUnit : SUnits) {
    if (!SUnit.isScheduled) {
      if (SUnit.NumPreds == 0 && SUnit.NumSuccs == 0) {
        ++DeadNodes;
        continue;
      }
      if (!AnyNotSched)
        dbgs() << "*** Scheduling failed! ***\n";
      dumpNode(SUnit);
      dbgs() << "has not been scheduled!\n";
      AnyNotSched = true;
    }
    if (SUnit.isScheduled &&
        (isBottomUp ? SUnit.getHeight() : SUnit.getDepth()) >
          unsigned(std::numeric_limits<int>::max())) {
      if (!AnyNotSched)
        dbgs() << "*** Scheduling failed! ***\n";
      dumpNode(SUnit);
      dbgs() << "has an unexpected "
           << (isBottomUp ? "Height" : "Depth") << " value!\n";
      AnyNotSched = true;
    }
    if (isBottomUp) {
      if (SUnit.NumSuccsLeft != 0) {
        if (!AnyNotSched)
          dbgs() << "*** Scheduling failed! ***\n";
        dumpNode(SUnit);
        dbgs() << "has successors left!\n";
        AnyNotSched = true;
      }
    } else {
      if (SUnit.NumPredsLeft != 0) {
        if (!AnyNotSched)
          dbgs() << "*** Scheduling failed! ***\n";
        dumpNode(SUnit);
        dbgs() << "has predecessors left!\n";
        AnyNotSched = true;
      }
    }
  }
  assert(!AnyNotSched);
  return SUnits.size() - DeadNodes;
}
#endif

// The topological order is maintained with the algorithm from "A dynamic
// topological sort algorithm for directed acyclic graphs" by David J. Pearce
// and Paul H.J. Kelly. The reachability tests follow "O'Reach: Even Faster
// Reachability in Large Graphs" by Kathrin Hanauer, Christian Schulz and
// Jonathan Trummer.
//
// InitDAGTopologicalSorting computes all the labels from scratch, and FixOrder
// applies the queued updates to them incrementally:
//
// * Order: inserting X->Y with Index(Y) < Index(X) moves the ancestors of X
//   that follow Y ahead of the descendants of Y that precede X, reusing their
//   indices. Nothing else moves. Removals never invalidate the order.
//
// * Levels and supportive vertices: a node's Level and ReachedBy only depend on
//   its predecessors' (and RLevel and Reaches on its successors'), so they are
//   recomputed in topological order, starting from the endpoints of the updated
//   edges and stopping wherever nothing changes. This doesn't need to know
//   whether an update really changed the edge set (e.g. addPred may find an
//   existing edge).
//
// * Cached rows: a path that an inserted edge X->Y completes reaches X, and
//   leaves Y, along edges that were already present; likewise for a path that
//   a removed edge breaks. So a row of descendants is only affected if its node
//   can reach X, and a row of ancestors if Y can reach its node. The order
//   decides that conservatively.

void ScheduleDAGTopologicalSort::newEpoch() {
  if (++Epoch != 0)
    return;
  // The counter wrapped around: clear the stale marks.
  for (NodeInfo &NI : Info)
    NI.FwdMark = NI.BwdMark = 0;
  Epoch = 1;
}

void ScheduleDAGTopologicalSort::allocate(unsigned N, int Index) {
  Info[N].Index = Index;
  Index2Node[Index] = N;
}

void ScheduleDAGTopologicalSort::InitDAGTopologicalSorting() {
  // Cancel pending updates, mark as valid.
  Dirty = false;
  Updates.clear();

  unsigned DAGSize = SUnits.size();
  Index2Node.resize(DAGSize);
  Info.assign(DAGSize, NodeInfo());
  Epoch = 0;

  // Number the nodes from the sinks up, using the Index fields as scratch
  // space for the number of successors not numbered yet.
  FwdStack.clear();
  for (SUnit &SU : SUnits) {
    assert(SU.NodeNum < DAGSize && &SUnits[SU.NodeNum] == &SU &&
           "NodeNum must index SUnits");
    unsigned Degree = count_if(SU.Succs, [&](const SDep &SuccDep) {
      return inDAG(SuccDep.getSUnit());
    });
    Info[SU.NodeNum].Index = Degree;
    if (Degree == 0)
      FwdStack.push_back(SU.NodeNum);
  }

  int Id = DAGSize;
  while (!FwdStack.empty()) {
    unsigned N = FwdStack.back();
    FwdStack.pop_back();
    allocate(N, --Id);
    for (const SDep &PredDep : SUnits[N].Preds) {
      const SUnit *Pred = PredDep.getSUnit();
      // If all successors of the node are numbered already, then the node can
      // be numbered now.
      if (inDAG(Pred) && !--Info[Pred->NodeNum].Index)
        FwdStack.push_back(Pred->NodeNum);
    }
  }
  assert(Id == 0 && "DAG contains a cycle");

  chooseSupportiveVertices();
  for (unsigned Idx = 0; Idx != DAGSize; ++Idx)
    recompute(Index2Node[Idx], /*Forward=*/true);
  for (unsigned Idx = DAGSize; Idx != 0; --Idx)
    recompute(Index2Node[Idx - 1], /*Forward=*/false);

  DescRows.resize(NumReachRows);
  AncRows.resize(NumReachRows);
  for (ReachRow &Row : DescRows)
    Row.reset(DAGSize);
  for (ReachRow &Row : AncRows)
    Row.reset(DAGSize);
  DescClock = AncClock = 0;

  NumTopoInits++;

#ifndef NDEBUG
  // Check correctness of the ordering
  for (SUnit &SU : SUnits)  {
    for (const SDep &PD : SU.Preds) {
      assert((!inDAG(PD.getSUnit()) ||
              Info[SU.NodeNum].Index > Info[PD.getSUnit()->NodeNum].Index) &&
             "Wrong topological sorting");
    }
  }
#endif
}

void ScheduleDAGTopologicalSort::chooseSupportiveVertices() {
  // A supportive vertex can only answer queries about pairs that straddle it,
  // so prefer nodes with many predecessors and many successors, like barriers.
  unsigned NumSupportive = std::min<unsigned>(TopoSupportiveVertices, 32);
  auto IsInDAG = [&](const SDep &Dep) { return inDAG(Dep.getSUnit()); };
  SmallVector<std::pair<uint64_t, unsigned>, 0> Candidates;
  for (const SUnit &SU : SUnits) {
    uint64_t Score =
        uint64_t(count_if(SU.Preds, IsInDAG)) * count_if(SU.Succs, IsInDAG);
    if (Score)
      Candidates.emplace_back(Score, SU.NodeNum);
  }
  if (Candidates.size() > NumSupportive) {
    std::nth_element(Candidates.begin(), Candidates.begin() + NumSupportive,
                     Candidates.end(), std::greater<>());
    Candidates.resize(NumSupportive);
  }
  for (auto [I, Candidate] : enumerate(Candidates))
    Info[Candidate.second].Self = 1u << I;
}

bool ScheduleDAGTopologicalSort::recompute(unsigned N, bool Forward) {
  NodeInfo &NI = Info[N];
  unsigned Level = 0;
  uint32_t Mask = NI.Self;
  for (const SDep &Dep : Forward ? SUnits[N].Preds : SUnits[N].Succs) {
    const SUnit *Other = Dep.getSUnit();
    if (!inDAG(Other))
      continue;
    const NodeInfo &OI = Info[Other->NodeNum];
    Level = std::max(Level, (Forward ? OI.Level : OI.RLevel) + 1);
    Mask |= Forward ? OI.ReachedBy : OI.Reaches;
  }
  unsigned &OldLevel = Forward ? NI.Level : NI.RLevel;
  uint32_t &OldMask = Forward ? NI.ReachedBy : NI.Reaches;
  if (Level == OldLevel && Mask == OldMask)
    return false;
  OldLevel = Level;
  OldMask = Mask;
  return true;
}

void ScheduleDAGTopologicalSort::propagate(ArrayRef<unsigned> Touched,
                                           bool Forward) {
  // Visit the affected nodes in topological order (reversed when going
  // backward), so that each is recomputed once, after everything it depends
  // on.
  newEpoch();
  Heap.clear();
  auto Push = [&](unsigned N) {
    NodeInfo &NI = Info[N];
    unsigned &Mark = Forward ? NI.FwdMark : NI.BwdMark;
    if (Mark == Epoch)
      return;
    Mark = Epoch;
    Heap.emplace_back(Forward ? NI.Index : -NI.Index, N);
    std::push_heap(Heap.begin(), Heap.end(), std::greater<>());
  };
  for (unsigned N : Touched)
    Push(N);
  while (!Heap.empty()) {
    std::pop_heap(Heap.begin(), Heap.end(), std::greater<>());
    unsigned N = Heap.back().second;
    Heap.pop_back();
    if (!recompute(N, Forward))
      continue;
    for (const SDep &Dep : Forward ? SUnits[N].Succs : SUnits[N].Preds)
      if (inDAG(Dep.getSUnit()))
        Push(Dep.getSUnit()->NodeNum);
  }
}

void ScheduleDAGTopologicalSort::insertIntoOrder(unsigned X, unsigned Y) {
  int Lower = Info[Y].Index;
  int Upper = Info[X].Index;
  // Is the order already consistent with X->Y?
  if (Lower >= Upper)
    return;

  // Collect the descendants of Y that precede X, and the ancestors of X that
  // follow Y. Edges leaving the window are either already consistent with the
  // order, or belong to updates that haven't been applied yet.
  newEpoch();
  FwdSeen.assign(1, Y);
  Info[Y].FwdMark = Epoch;
  for (unsigned I = 0; I != FwdSeen.size(); ++I) {
    for (const SDep &SuccDep : SUnits[FwdSeen[I]].Succs) {
      const SUnit *Succ = SuccDep.getSUnit();
      if (!inDAG(Succ))
        continue;
      NodeInfo &SI = Info[Succ->NodeNum];
      assert(SI.Index != Upper && "Inserted edge creates a loop!");
      if (SI.Index > Lower && SI.Index < Upper && SI.FwdMark != Epoch) {
        SI.FwdMark = Epoch;
        FwdSeen.push_back(Succ->NodeNum);
      }
    }
  }
  BwdSeen.assign(1, X);
  Info[X].BwdMark = Epoch;
  for (unsigned I = 0; I != BwdSeen.size(); ++I) {
    for (const SDep &PredDep : SUnits[BwdSeen[I]].Preds) {
      const SUnit *Pred = PredDep.getSUnit();
      if (!inDAG(Pred))
        continue;
      NodeInfo &PI = Info[Pred->NodeNum];
      if (PI.Index > Lower && PI.Index < Upper && PI.BwdMark != Epoch) {
        PI.BwdMark = Epoch;
        BwdSeen.push_back(Pred->NodeNum);
      }
    }
  }

  // Reuse the indices of both sets, placing the ancestors of X first. Each set
  // keeps its internal order.
  auto ByIndex = [&](unsigned A, unsigned B) {
    return Info[A].Index < Info[B].Index;
  };
  llvm::sort(FwdSeen, ByIndex);
  llvm::sort(BwdSeen, ByIndex);
  SmallVector<int, 32> Indices;
  for (unsigned N : BwdSeen)
    Indices.push_back(Info[N].Index);
  for (unsigned N : FwdSeen)
    Indices.push_back(Info[N].Index);
  std::inplace_merge(Indices.begin(), Indices.begin() + BwdSeen.size(),
                     Indices.end());
  unsigned I = 0;
  for (unsigned N : BwdSeen)
    allocate(N, Indices[I++]);
  for (unsigned N : FwdSeen)
    allocate(N, Indices[I++]);
}

void ScheduleDAGTopologicalSort::invalidateRows(const Update &U) {
  int PredIndex = Info[U.Pred->NodeNum].Index;
  int SuccIndex = Info[U.Succ->NodeNum].Index;
  for (ReachRow &Row : DescRows)
    if (Row.Node >= 0 && Info[Row.Node].Index <= PredIndex)
      (U.IsInsert ? Row.NegValid : Row.PosValid) = false;
  for (ReachRow &Row : AncRows)
    if (Row.Node >= 0 && Info[Row.Node].Index >= SuccIndex)
      (U.IsInsert ? Row.NegValid : Row.PosValid) = false;
}

void ScheduleDAGTopologicalSort::FixOrder() {
  // Recompute from scratch after new nodes have been added.
  if (Dirty) {
    InitDAGTopologicalSorting();
    return;
  }
  if (Updates.empty())
    return;

  erase_if(Updates,
           [&](const Update &U) { return !inDAG(U.Pred) || !inDAG(U.Succ); });

  // Restore the order first; everything below relies on it.
  for (const Update &U : Updates)
    if (U.IsInsert)
      insertIntoOrder(U.Pred->NodeNum, U.Succ->NodeNum);

  SmallVector<unsigned, 16> FwdTouched, BwdTouched;
  for (const Update &U : Updates) {
    invalidateRows(U);
    FwdTouched.push_back(U.Succ->NodeNum);
    BwdTouched.push_back(U.Pred->NodeNum);
  }
  Updates.clear();

  propagate(FwdTouched, /*Forward=*/true);
  propagate(BwdTouched, /*Forward=*/false);
}

void ScheduleDAGTopologicalSort::queueUpdate(SUnit *Pred, SUnit *Succ,
                                             bool IsInsert) {
  Dirty = Dirty || Updates.size() > MaxQueuedUpdates;
  if (Dirty)
    return;
  Updates.push_back({Pred, Succ, IsInsert});
}

void ScheduleDAGTopologicalSort::AddPredQueued(SUnit *Y, SUnit *X) {
  queueUpdate(X, Y, /*IsInsert=*/true);
  NumNewPredsAdded++;
}

void ScheduleDAGTopologicalSort::AddPred(SUnit *Y, SUnit *X) {
  AddPredQueued(Y, X);
}

void ScheduleDAGTopologicalSort::RemovePred(SUnit *M, SUnit *N) {
  queueUpdate(N, M, /*IsInsert=*/false);
}

std::vector<int> ScheduleDAGTopologicalSort::GetSubGraph(const SUnit &StartSU,
                                                         const SUnit &TargetSU,
                                                         bool &Success) {
  FixOrder();
  int LowerBound = Info[StartSU.NodeNum].Index;
  int UpperBound = Info[TargetSU.NodeNum].Index;
  bool Found = false;
  std::vector<int> Nodes;

  if (LowerBound > UpperBound) {
    Success = false;
    return Nodes;
  }

  newEpoch();

  // Starting from StartSU, visit all successors up
  // to UpperBound.
  FwdStack.assign(1, StartSU.NodeNum);
  do {
    unsigned N = FwdStack.back();
    FwdStack.pop_back();
    for (const SDep &SD : llvm::reverse(SUnits[N].Succs)) {
      const SUnit *Succ = SD.getSUnit();
      // Edges to non-SUnits are allowed but ignored (e.g. ExitSU).
      if (!inDAG(Succ))
        continue;
      NodeInfo &SI = Info[Succ->NodeNum];
      if (SI.Index == UpperBound) {
        Found = true;
        continue;
      }
      // Visit successors if not already and in affected region.
      if (SI.FwdMark != Epoch && SI.Index < UpperBound) {
        SI.FwdMark = Epoch;
        FwdStack.push_back(Succ->NodeNum);
      }
    }
  } while (!FwdStack.empty());

  if (!Found) {
    Success = false;
    return Nodes;
  }

  Found = false;

  // Starting from TargetSU, visit all predecessors up
  // to LowerBound. SUs that are visited by the two
  // passes are added to Nodes.
  BwdStack.assign(1, TargetSU.NodeNum);
  do {
    unsigned N = BwdStack.back();
    BwdStack.pop_back();
    for (const SDep &SD : llvm::reverse(SUnits[N].Preds)) {
      const SUnit *Pred = SD.getSUnit();
      // Edges to non-SUnits are allowed but ignored (e.g. EntrySU).
      if (!inDAG(Pred))
        continue;
      NodeInfo &PI = Info[Pred->NodeNum];
      if (PI.Index == LowerBound) {
        Found = true;
        continue;
      }
      if (PI.BwdMark != Epoch && PI.FwdMark == Epoch) {
        PI.BwdMark = Epoch;
        BwdStack.push_back(Pred->NodeNum);
        Nodes.push_back(Pred->NodeNum);
      }
    }
  } while (!BwdStack.empty());

  assert(Found && "Error in SUnit Graph!");
  Success = true;
  return Nodes;
}

struct ScheduleDAGTopologicalSort::Bounds {
  int MinIndex, MaxIndex;
  unsigned MinLevel, MaxLevel;
  unsigned MinRLevel, MaxRLevel;
  /// Supportive vertices that reach any (all) of the nodes, and that any (all)
  /// of the nodes reach.
  uint32_t AnyReachedBy, AllReachedBy;
  uint32_t AnyReaches, AllReaches;

  explicit Bounds(const NodeInfo &NI)
      : MinIndex(NI.Index), MaxIndex(NI.Index), MinLevel(NI.Level),
        MaxLevel(NI.Level), MinRLevel(NI.RLevel), MaxRLevel(NI.RLevel),
        AnyReachedBy(NI.ReachedBy), AllReachedBy(NI.ReachedBy),
        AnyReaches(NI.Reaches), AllReaches(NI.Reaches) {}

  void add(const NodeInfo &NI) {
    MinIndex = std::min(MinIndex, NI.Index);
    MaxIndex = std::max(MaxIndex, NI.Index);
    MinLevel = std::min(MinLevel, NI.Level);
    MaxLevel = std::max(MaxLevel, NI.Level);
    MinRLevel = std::min(MinRLevel, NI.RLevel);
    MaxRLevel = std::max(MaxRLevel, NI.RLevel);
    AnyReachedBy |= NI.ReachedBy;
    AllReachedBy &= NI.ReachedBy;
    AnyReaches |= NI.Reaches;
    AllReaches &= NI.Reaches;
  }
};

ScheduleDAGTopologicalSort::Bounds
ScheduleDAGTopologicalSort::bounds(ArrayRef<const SUnit *> SUs) const {
  assert(!SUs.empty() && all_of(SUs, [&](const SUnit *SU) {
    return inDAG(SU);
  }) && "Expected nodes in the DAG");
  Bounds B(Info[SUs.front()->NodeNum]);
  for (const SUnit *SU : SUs.drop_front())
    B.add(Info[SU->NodeNum]);
  return B;
}

ScheduleDAGTopologicalSort::Reach
ScheduleDAGTopologicalSort::classify(const Bounds &From, const Bounds &To) {
  // Every edge increases Index and Level, and decreases RLevel.
  if (From.MinIndex >= To.MaxIndex || From.MinLevel >= To.MaxLevel ||
      From.MaxRLevel <= To.MinRLevel)
    return Reach::No;
  // Some node in From reaches a supportive vertex that reaches some node in To.
  if (From.AnyReaches & To.AnyReachedBy)
    return Reach::Yes;
  // A supportive vertex reaches all of From but none of To, or all of To reach
  // a supportive vertex that none of From reaches.
  if ((From.AllReachedBy & ~To.AnyReachedBy) ||
      (To.AllReaches & ~From.AnyReaches))
    return Reach::No;
  return Reach::Unknown;
}

ScheduleDAGTopologicalSort::Reach
ScheduleDAGTopologicalSort::lookup(const ReachRow &Row, unsigned N) {
  if (Row.PosValid && Row.Pos.test(N))
    return Reach::Yes;
  if (Row.NegValid && Row.Neg.test(N))
    return Reach::No;
  return Reach::Unknown;
}

ScheduleDAGTopologicalSort::ReachRow &
ScheduleDAGTopologicalSort::getRow(MutableArrayRef<ReachRow> Rows,
                                   unsigned &Clock, int NodeInfo::*RowOf,
                                   unsigned N) {
  if (int Idx = Info[N].*RowOf; Idx >= 0) {
    Rows[Idx].Referenced = true;
    return Rows[Idx];
  }
  // Give each recently used row a second chance before evicting it.
  while (true) {
    unsigned Idx = Clock;
    Clock = (Clock + 1) % Rows.size();
    ReachRow &Row = Rows[Idx];
    if (Row.Node >= 0 && Row.Referenced) {
      Row.Referenced = false;
      continue;
    }
    if (Row.Node >= 0)
      Info[Row.Node].*RowOf = -1;
    Row.Node = N;
    Row.Referenced = true;
    Row.PosValid = Row.NegValid = false;
    Info[N].*RowOf = Idx;
    return Row;
  }
}

bool ScheduleDAGTopologicalSort::reaches(ArrayRef<const SUnit *> From,
                                         ArrayRef<const SUnit *> To) {
  ++NumReachQueries;
  Bounds FromB = bounds(From), ToB = bounds(To);
  // Cached rows only describe single nodes.
  const SUnit *Src = From.size() == 1 ? From.front() : nullptr;
  const SUnit *Dst = To.size() == 1 ? To.front() : nullptr;

  Reach Answer = classify(FromB, ToB);
  if (Answer == Reach::Unknown && Src && Dst) {
    if (int Row = Info[Src->NodeNum].DescRow; Row >= 0) {
      DescRows[Row].Referenced = true;
      Answer = lookup(DescRows[Row], Dst->NodeNum);
    }
    if (int Row = Info[Dst->NodeNum].AncRow;
        Answer == Reach::Unknown && Row >= 0) {
      AncRows[Row].Referenced = true;
      Answer = lookup(AncRows[Row], Src->NodeNum);
    }
  }
  if (Answer != Reach::Unknown)
    return Answer == Reach::Yes;

  // Fall back to a bidirectional search, applying the same tests at every node
  // it reaches, and recording what it learns.
  ++NumReachSearches;
  ReachRow *SrcRow =
      Src ? &getRow(DescRows, DescClock, &NodeInfo::DescRow, Src->NodeNum)
          : nullptr;
  ReachRow *DstRow =
      Dst ? &getRow(AncRows, AncClock, &NodeInfo::AncRow, Dst->NodeNum)
          : nullptr;

  // Can a node reached by the forward search reach To?
  auto ClassifyFwd = [&](unsigned N) {
    const NodeInfo &NI = Info[N];
    Reach R = classify(Bounds(NI), ToB);
    if (R == Reach::Unknown && DstRow)
      R = lookup(*DstRow, N);
    if (R == Reach::Unknown && Dst && NI.DescRow >= 0)
      R = lookup(DescRows[NI.DescRow], Dst->NodeNum);
    return R;
  };
  // Can From reach a node reached by the backward search?
  auto ClassifyBwd = [&](unsigned N) {
    const NodeInfo &NI = Info[N];
    Reach R = classify(FromB, Bounds(NI));
    if (R == Reach::Unknown && SrcRow)
      R = lookup(*SrcRow, N);
    if (R == Reach::Unknown && Src && NI.AncRow >= 0)
      R = lookup(AncRows[NI.AncRow], Src->NodeNum);
    return R;
  };

  newEpoch();
  FwdStack.clear();
  BwdStack.clear();
  FwdSeen.clear();
  BwdSeen.clear();
  bool Found = false;
  for (const SUnit *SU : From) {
    unsigned N = SU->NodeNum;
    if (Info[N].FwdMark == Epoch)
      continue;
    Info[N].FwdMark = Epoch;
    FwdSeen.push_back(N);
    Reach R = ClassifyFwd(N);
    Found |= R == Reach::Yes;
    if (R == Reach::Unknown)
      FwdStack.push_back(N);
  }
  for (const SUnit *SU : To) {
    unsigned N = SU->NodeNum;
    assert(Info[N].FwdMark != Epoch && "Expected disjoint sets");
    if (Info[N].BwdMark == Epoch)
      continue;
    Info[N].BwdMark = Epoch;
    BwdSeen.push_back(N);
    Reach R = ClassifyBwd(N);
    Found |= R == Reach::Yes;
    if (R == Reach::Unknown)
      BwdStack.push_back(N);
  }

  // Expand the smaller frontier at each step. If either runs dry, nothing it
  // reached can reach (or be reached from) the other side.
  bool FwdExhausted = false, BwdExhausted = false;
  while (!Found) {
    if (FwdStack.empty()) {
      FwdExhausted = true;
      break;
    }
    if (BwdStack.empty()) {
      BwdExhausted = true;
      break;
    }
    bool Forward = FwdStack.size() <= BwdStack.size();
    std::vector<unsigned> &Stack = Forward ? FwdStack : BwdStack;
    unsigned N = Stack.back();
    Stack.pop_back();
    for (const SDep &Dep : Forward ? SUnits[N].Succs : SUnits[N].Preds) {
      const SUnit *Next = Dep.getSUnit();
      if (!inDAG(Next))
        continue;
      NodeInfo &NI = Info[Next->NodeNum];
      unsigned &Mark = Forward ? NI.FwdMark : NI.BwdMark;
      if (Mark == Epoch)
        continue;
      // Have the two searches met?
      if ((Forward ? NI.BwdMark : NI.FwdMark) == Epoch) {
        Found = true;
        break;
      }
      Mark = Epoch;
      (Forward ? FwdSeen : BwdSeen).push_back(Next->NodeNum);
      Reach R =
          Forward ? ClassifyFwd(Next->NodeNum) : ClassifyBwd(Next->NodeNum);
      if (R == Reach::Yes) {
        Found = true;
        break;
      }
      if (R == Reach::Unknown)
        Stack.push_back(Next->NodeNum);
    }
  }

  // Everything the forward search reached is reachable from Src, and
  // everything the backward search reached reaches Dst.
  if (SrcRow) {
    BitVector &Pos = SrcRow->validPos();
    for (unsigned N : drop_begin(FwdSeen))
      Pos.set(N);
    if (BwdExhausted) {
      BitVector &Neg = SrcRow->validNeg();
      for (unsigned N : BwdSeen)
        Neg.set(N);
    }
  }
  if (DstRow) {
    BitVector &Pos = DstRow->validPos();
    for (unsigned N : drop_begin(BwdSeen))
      Pos.set(N);
    if (FwdExhausted) {
      BitVector &Neg = DstRow->validNeg();
      for (unsigned N : FwdSeen)
        Neg.set(N);
    }
  }
  if (SrcRow && DstRow) {
    (Found ? SrcRow->validPos() : SrcRow->validNeg()).set(Dst->NodeNum);
    (Found ? DstRow->validPos() : DstRow->validNeg()).set(Src->NodeNum);
  }
  return Found;
}

bool ScheduleDAGTopologicalSort::IsReachable(const SUnit *SU,
                                             const SUnit *TargetSU) {
  assert(TargetSU != nullptr && "Invalid target SUnit");
  assert(SU != nullptr && "Invalid SUnit");
  if (SU == TargetSU)
    return false;
  FixOrder();
  return reaches(TargetSU, SU);
}

bool ScheduleDAGTopologicalSort::IsReachable(
    ArrayRef<const SUnit *> SUs, ArrayRef<const SUnit *> TargetSUs) {
  if (SUs.empty() || TargetSUs.empty())
    return false;
  FixOrder();
  return reaches(TargetSUs, SUs);
}

bool ScheduleDAGTopologicalSort::WillCreateCycle(SUnit *TargetSU, SUnit *SU) {
  // Is SU reachable from TargetSU, or from any of the nodes that TargetSU has
  // assigned register dependences on?
  SmallVector<const SUnit *, 4> From;
  if (TargetSU != SU)
    From.push_back(TargetSU);
  for (const SDep &PredDep : TargetSU->Preds)
    if (PredDep.isAssignedRegDep() && PredDep.getSUnit() != SU)
      From.push_back(PredDep.getSUnit());
  const SUnit *To = SU;
  return IsReachable(To, From);
}

void ScheduleDAGTopologicalSort::AddSUnitWithoutPredecessors(const SUnit *SU) {
  assert(SU->NodeNum == Index2Node.size() && "Node cannot be added at the end");
  assert(SU->NumPreds == 0 && "Can only add SU's with no predecessors");
  Info.emplace_back();
  Info.back().Index = Index2Node.size();
  Index2Node.push_back(SU->NodeNum);
  for (ReachRow &Row : DescRows) {
    Row.Pos.resize(Info.size());
    Row.Neg.resize(Info.size());
  }
  for (ReachRow &Row : AncRows) {
    Row.Pos.resize(Info.size());
    Row.Neg.resize(Info.size());
  }
}

ScheduleDAGTopologicalSort::ScheduleDAGTopologicalSort(
    std::vector<SUnit> &sunits, SUnit * /*exitsu*/)
    : SUnits(sunits) {}

ScheduleHazardRecognizer::~ScheduleHazardRecognizer() = default;
