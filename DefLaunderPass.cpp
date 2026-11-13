#include "DefLaunderPass.h"

#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constant.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/IntrinsicInst.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/CommandLine.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <utility>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> DLEnable("obf-dl", llvm::cl::init(true), llvm::cl::desc("Hoist Values Upwards through basic blocks, Then reallocates them in each block to make them hard to track"));

static constexpr unsigned Density = 100;
static constexpr unsigned MinBitwidth = 32;
static constexpr unsigned MaxBitwidth = 64;
static constexpr bool SkipConstants = false;
static constexpr bool SkipArgs = false;
static constexpr bool SkipPhiV = false;
static constexpr bool SkipLoopDef = false;

static bool HasAnnotationTag(Function& f, const char* tag)
{
    Module* m = f.getParent();
    GlobalVariable* ga = m->getGlobalVariable("llvm.global.annotations");
    if (!ga || !ga->hasInitializer()) return false;
    auto* ca = dyn_cast<ConstantArray>(ga->getOperand(0));
    if (!ca) return false;
    for (unsigned i = 0; i < ca->getNumOperands(); ++i)
    {
        auto* cs = dyn_cast<ConstantStruct>(ca->getOperand(i));
        if (!cs || cs->getNumOperands() < 2) continue;
        if (cs->getOperand(0)->stripPointerCasts() != &f) continue;
        auto* annogv = dyn_cast<GlobalVariable>(cs->getOperand(1)->stripPointerCasts());
        if (!annogv || !annogv->hasInitializer()) continue;
        auto* cds = dyn_cast<ConstantDataSequential>(annogv->getInitializer());
        if (!cds || !cds->isString()) continue;
        if (strcmp(cds->getAsString().str().c_str(), tag) == 0) return true;
    }
    return false;
}

static bool HasObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "obfus"); }
static bool HasNoObfuscateAnnotation(Function& f) { return HasAnnotationTag(f, "noobfus"); }

static std::mt19937_64& DLRng()
{
    static std::mt19937_64 rng;
    static bool seeded = false;
    if (!seeded)
    {
        std::random_device rd;
        rng.seed(((uint64_t)rd() << 32) ^ (uint64_t)rd());
        seeded = true;
    }
    return rng;
}

static BasicBlock* DefBBOf(Value* v, Function& f)
{
    if (!v) return nullptr;
    if (auto* i = dyn_cast<Instruction>(v)) return i->getParent();
    if (isa<Argument>(v)) return &f.getEntryBlock();
    if (isa<Constant>(v)) return &f.getEntryBlock();
    return nullptr;
}

static bool IsEligibleV(Value* v)
{
    if (!v) return false;
    auto* it = dyn_cast<IntegerType>(v->getType());
    if (!it) return false;
    const unsigned bw = it->getBitWidth();
    if (bw < MinBitwidth) return false;
    if (bw > MaxBitwidth) return false;

    if (auto* i = dyn_cast<Instruction>(v))
    {
        // a terminator (br, ret, switch, invoke, resume, catchswitch, catchret, cleanupret, indirectbr) returns no SSA value a user can read, so there is nothing to launder; the isTerminator check must come before the PHI check since catchswitch is both
        if (i->isTerminator()) return false;
        if (isa<PHINode>(i) && SkipPhiV) return false;
        return true;
    }
    if (isa<Argument>(v)) return !SkipArgs;
    if (isa<ConstantInt>(v)) return !SkipConstants;
    if (isa<ConstantExpr>(v)) return !SkipConstants;
    return false;
}

static bool IsEligibleUser(Instruction& i, const LoopInfo& li)
{
    // PHIs require per predecessor edge load placement (one load in each incoming edge) and critical edges force CFG splits to make that legal; outside the current scope so PHI users are skipped wholesale
    if (isa<PHINode>(&i)) return false;
    // intrinsic operand slots have strict type constraints (metadata, immarg, specific-type) that a plain volatile load may not satisfy; skipping intrinsics avoids verifier failures from laundering e.g. llvm.assume, llvm.lifetime.start, llvm.memcpy length
    if (isa<IntrinsicInst>(&i)) return false;
    if (li.getLoopFor(i.getParent()) != nullptr) return false;
    return true;
}

struct DLSlot
{
    Instruction* User = nullptr;
    unsigned OpIdx = 0;
};

struct DLSitePick
{
    Value* V = nullptr;
    unsigned Bw = 0;
};

static std::vector<DLSitePick> CollectAllEligibleVs(Function& f, const LoopInfo& li)
{
    std::vector<DLSitePick> out;
    SmallPtrSet<Value*, 32> seen;
    for (BasicBlock& bb : f)
    {
        if (li.getLoopFor(&bb) != nullptr) continue;
        for (Instruction& i : bb)
        {
            if (!IsEligibleUser(i, li)) continue;
            for (unsigned opidx = 0, n = i.getNumOperands(); opidx < n; ++opidx)
            {
                // SwitchInst case value slots (ops 2, 4, ...) MUST be ConstantInt by LLVM IR invariants, and GEP indices beyond op 0 include struct field positions that also must stay ConstantInt (walking the source type per op to tell them apart is non trivial, so the safe choice is to skip all GEP indices)
                if (isa<SwitchInst>(&i) && opidx != 0) continue;
                if (isa<GetElementPtrInst>(&i) && opidx != 0) continue;

                Value* v = i.getOperand(opidx);
                if (!IsEligibleV(v)) continue;

                BasicBlock* defbb = nullptr;
                if (auto* vi = dyn_cast<Instruction>(v))
                {
                    defbb = vi->getParent();
                    if (defbb == &bb) continue;
                    // a none PHI Instruction defined inside a loop runs once per iteration; a volatile store at V->getNextNode() captures the current-iter value, but non-loop readers would see the LAST iter's value when the SSA value is the live out of a specific iter; undefined in general. PHI V's in loop headers are a controlled case: the loop header's post-PHI store is the LCSSA equivalent read, so they stay unless SkipLoopDef is on
                    if (li.getLoopFor(defbb) != nullptr)
                    {
                        const bool isphi = isa<PHINode>(vi);
                        if (!isphi) continue;
                        if (SkipLoopDef) continue;
                    }
                }
                else
                {
                    defbb = &f.getEntryBlock();
                    if (defbb == &bb) continue;
                }

                if (!seen.insert(v).second) continue;
                const unsigned bw = v->getType()->getIntegerBitWidth();
                out.push_back({v, bw});
            }
        }
    }
    return out;
}

static std::vector<DLSlot> CollectAllSlotsForV(Function& f, const LoopInfo& li, Value* v, BasicBlock* defbb)
{
    std::vector<DLSlot> out;
    for (BasicBlock& bb : f)
    {
        if (&bb == defbb) continue;
        if (li.getLoopFor(&bb) != nullptr) continue;
        for (Instruction& i : bb)
        {
            if (isa<PHINode>(&i)) continue;
            if (isa<IntrinsicInst>(&i)) continue;
            for (unsigned opidx = 0, n = i.getNumOperands(); opidx < n; ++opidx)
            {
                // mirror the collection time Switch/GEP guards; a slot we rewrite here must also have been eligible at collection time, otherwise we'd overwrite a required Constant operand with a load and fail IR verification
                if (isa<SwitchInst>(&i) && opidx != 0) continue;
                if (isa<GetElementPtrInst>(&i) && opidx != 0) continue;
                if (i.getOperand(opidx) == v)
                    out.push_back({&i, opidx});
            }
        }
    }
    return out;
}

static bool EmitForV(Function& f, const DLSitePick& pick, const LoopInfo& li)
{
    LLVMContext& ctx = f.getContext();
    IntegerType* vt = IntegerType::get(ctx, pick.Bw);
    BasicBlock& entry = f.getEntryBlock();

    BasicBlock* defbb = DefBBOf(pick.V, f);
    if (!defbb) return false;

    std::vector<DLSlot> slots = CollectAllSlotsForV(f, li, pick.V, defbb);
    if (slots.empty()) return false;

    IRBuilder<> allocab(&entry, entry.getFirstNonPHIIt());
    AllocaInst* slotalloca = allocab.CreateAlloca(vt, nullptr, "DLSlot");

    StoreInst* st = nullptr;
    if (auto* vi = dyn_cast<Instruction>(pick.V))
    {
        if (isa<PHINode>(vi))
        {
            // PHIs must stay contiguous at the top of their BB, so the store can't go immediately after the PHI; using getFirstInsertionPt() lands past all PHIs and any EH pad, which is the earliest legal position in the def BB. In a loop header this fires once per iteration; correct for non-loop readers which see the last-iter value
            BasicBlock* vbb = vi->getParent();
            IRBuilder<> storeb(vbb, vbb->getFirstInsertionPt());
            st = storeb.CreateStore(pick.V, slotalloca);
        }
        else
        {
            Instruction* afterv = vi->getNextNode();
            // V->getNextNode() returns null only when V IS the terminator, which IsEligibleV already rejects; the guard here catches a corruption we'd rather fail cleanly on than crash in the store builder
            if (!afterv) { slotalloca->eraseFromParent(); return false; }
            IRBuilder<> storeb(afterv);
            st = storeb.CreateStore(pick.V, slotalloca);
        }
    }
    else
    {
        // for Argument and Constant V the def is logically at function entry; reuse allocab (its insertion point still sits just past the alloca we created) so the store lands with the alloca without rescanning entry for an insertion point
        st = allocab.CreateStore(pick.V, slotalloca);
    }
    st->setVolatile(true);

    // MapVector preserves insertion order, so load emission order across use BBs is deterministic across builds; a plain DenseMap would hash on BasicBlock* and surface a none deterministic ordering in both the IR and any downstream diff
    MapVector<BasicBlock*, std::vector<DLSlot>> bybb;
    for (const DLSlot& sl : slots)
        bybb[sl.User->getParent()].push_back(sl);

    for (auto& kv : bybb)
    {
        BasicBlock* bb = kv.first;
        auto& bbslots = kv.second;

        IRBuilder<> loadb(bb, bb->getFirstNonPHIIt());
        LoadInst* ld = loadb.CreateLoad(vt, slotalloca, /*isVolatile=*/true, "DLLoad");

        for (const DLSlot& sl : bbslots)
            sl.User->setOperand(sl.OpIdx, ld);
    }

    return true;
}

static bool ProcessFunction(Function& f)
{
    if (f.isDeclaration()) return false;
    if (!HasObfuscateAnnotation(f)) return false;
    if (HasNoObfuscateAnnotation(f)) return false;

    DominatorTree dt;
    dt.recalculate(f);
    LoopInfo li;
    li.analyze(dt);

    std::vector<DLSitePick> vs = CollectAllEligibleVs(f, li);

    if (Density < 100 && !vs.empty())
    {
        auto& rng = DLRng();
        size_t write = 0;
        for (size_t read = 0; read < vs.size(); ++read)
        {
            if ((unsigned)(rng() % 100) < Density)
            {
                if (write != read) vs[write] = vs[read];
                ++write;
            }
        }
        vs.resize(write);
    }

    bool mutated = false;
    for (const DLSitePick& pick : vs)
    {
        if (EmitForV(f, pick, li)) mutated = true;
    }
    return mutated;
}

bool RunDefLaunder(Module& m)
{
    if (!DLEnable) return false;
    bool mutated = false;
    for (Function& f : m) mutated |= ProcessFunction(f);
    return mutated;
}

PreservedAnalyses DefLaunderPass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunDefLaunder(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}