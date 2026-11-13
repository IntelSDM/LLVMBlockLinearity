#include "CrossBlockSubstratePass.h"

#include "mbagen/PolyMBA.h"
#include "mbagen/PolyMBACatalog.h"

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Analysis/LoopInfo.h"
#include "llvm/IR/Argument.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/CFG.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Dominators.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/InlineAsm.h"
#include "llvm/IR/InstrTypes.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/Intrinsics.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Metadata.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Transforms/Utils/BasicBlockUtils.h"

#include <cstdint>
#include <cstring>
#include <random>
#include <string>
#include <utility>
#include <vector>

using namespace llvm;

llvm::cl::opt<bool> CBSPEnable("obf-cbsp", llvm::cl::init(true), llvm::cl::desc("Load Bares Blocks Using Horner Encode/Decode For Constants Through Blocks"));

static constexpr bool PhiEnable = true;
static constexpr unsigned PhiMinPreds = 2;
static constexpr bool EscalateEnable = true;
static constexpr bool TagHornerEmits = true;
static constexpr bool HoistLoopMorphs = true;
static constexpr bool ArxBranchEnable = true;
static constexpr unsigned ArxRot = 13;
static constexpr unsigned CatalogCount = 16;
static constexpr unsigned CatalogInverseCapD = 80;
static constexpr unsigned CatalogVerifyBits = 18;
static constexpr unsigned CatalogVerifyRandom = 10000;
static constexpr unsigned CatalogA2Valuation = 0;
static constexpr unsigned CatalogMaxTriesMult = 8;
static constexpr size_t AuxPoolCap = 64;

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

static std::mt19937_64& CbspRng()
{
    static std::mt19937_64 rng;
    static bool seeded = false;
    if (!seeded)
    {
        std::random_device rd;
        uint64_t seed = ((uint64_t)rd() << 32) | rd();
        rng.seed(seed);
        seeded = true;
    }
    return rng;
}

static std::mt19937_64& CbspArxRng()
{
    static std::mt19937_64 rng;
    static bool seeded = false;
    if (!seeded)
    {
        std::random_device rd;
        uint64_t seed = ((uint64_t)rd() << 32) ^ (uint64_t)rd();
        rng.seed(seed);
        seeded = true;
    }
    return rng;
}

static uint64_t MaskW(uint64_t v, unsigned bw)
{
    if (bw >= 64) return v;
    return v & ((uint64_t(1) << bw) - 1);
}

static Value* InsertFoldBarrier(IRBuilder<>& b, Value* v)
{
    auto* vt = cast<IntegerType>(v->getType());
    FunctionType* ft = FunctionType::get(vt, {vt}, false);
    InlineAsm* ia = InlineAsm::get(ft, "", "=r,0", /*hasSideEffects*/true);
    return b.CreateCall(ia, {v}, "cbsp.barr");
}

static constexpr const char* kEscalateKind = "mbagen.escalate";
static constexpr const char* kEscalateValue = "poly_const";

static bool IsEscalatedSite(const Instruction* i)
{
    if (!i) return false;
    MDNode* md = i->getMetadata(kEscalateKind);
    if (!md) return false;
    for (unsigned k = 0, n = md->getNumOperands(); k < n; ++k)
    {
        if (auto* ms = dyn_cast<MDString>(md->getOperand(k).get()))
            if (ms->getString() == kEscalateValue) return true;
    }
    return false;
}

static void TagMbagenEscalate(Instruction* i)
{
    if (!i) return;
    LLVMContext& ctx = i->getContext();
    MDString* tag = MDString::get(ctx, kEscalateValue);
    Metadata* ops[] = { tag };
    MDNode* node = MDNode::get(ctx, ArrayRef<Metadata*>(ops));
    i->setMetadata(kEscalateKind, node);
}

struct SitePick
{
    Instruction* user = nullptr;
    unsigned opidx = 0;
    ConstantInt* k = nullptr;
    unsigned bw = 0;
    bool escalated = false;
};

static bool IsEligibleUser(Instruction& i, const BasicBlock* entrybb, const LoopInfo& li)
{
    if (!isa<BinaryOperator>(&i)) return false;
    BasicBlock* bb = i.getParent();
    if (bb == entrybb) return false;
    if (li.getLoopFor(bb) != nullptr) return false;
    return true;
}

static SitePick FindFirstSite(Function& f, const LoopInfo& li)
{
    BasicBlock* entrybb = &f.getEntryBlock();
    SitePick first{};
    SitePick firstesc{};
    for (BasicBlock& bb : f)
    {
        if (&bb == entrybb) continue;
        if (li.getLoopFor(&bb) != nullptr) continue;
        for (Instruction& i : bb)
        {
            if (!IsEligibleUser(i, entrybb, li)) continue;
            const bool usertag = EscalateEnable && IsEscalatedSite(&i);
            for (unsigned opidx = 0, n = i.getNumOperands(); opidx < n; ++opidx)
            {
                ConstantInt* k = dyn_cast_or_null<ConstantInt>(i.getOperand(opidx));
                if (!k) continue;
                unsigned bw = k->getBitWidth();
                if (bw != 32 && bw != 64) continue;
                SitePick cand{&i, opidx, k, bw, usertag};
                if (!first.user) first = cand;
                if (usertag && !firstesc.user) firstesc = cand;
            }
        }
    }
    // escalated sites take priority over the first eligible site, so a function with any poly_const-tagged binop gets its chain treatment there regardless of textual order
    if (firstesc.user) return firstesc;
    return first;
}

static BasicBlock* ResolveLoopHoistTarget(const LoopInfo& li, const DominatorTree& dt, BasicBlock* l, BasicBlock* usebb)
{
    Loop* loop = li.getLoopFor(l);
    if (!loop) return nullptr;
    Loop* outer = loop;
    while (Loop* parent = outer->getParentLoop()) outer = parent;
    BasicBlock* ph = outer->getLoopPreheader();
    if (!ph) return nullptr;
    // preheader itself must be loop free, else its enclosing loop would reexecute the morph each iteration and desync the slot encoding
    if (li.getLoopFor(ph) != nullptr) return nullptr;
    // ph must dominate use.BB; true by construction (ph dominates loop header, which dominates L, which dominates use.BB), but a defensive check costs nothing and catches malformed loop info
    if (!dt.dominates(ph, usebb)) return nullptr;
    return ph;
}

static std::vector<BasicBlock*> CollectChain(const DominatorTree& dt, const LoopInfo& li, BasicBlock* usebb, BasicBlock* entrybb)
{
    std::vector<BasicBlock*> bottomup;
    SmallPtrSet<BasicBlock*, 16> seen;
    DomTreeNode* node = dt.getNode(usebb);
    if (!node) return bottomup;
    while (true)
    {
        DomTreeNode* idom = node->getIDom();
        if (!idom) break;
        BasicBlock* bb = idom->getBlock();
        if (bb == entrybb) break;
        if (li.getLoopFor(bb) != nullptr)
        {
            if (HoistLoopMorphs)
            {
                if (BasicBlock* ph = ResolveLoopHoistTarget(li, dt, bb, usebb))
                    if (seen.insert(ph).second) bottomup.push_back(ph);
            }
        }
        else
        {
            if (seen.insert(bb).second) bottomup.push_back(bb);
        }
        node = idom;
    }
    std::reverse(bottomup.begin(), bottomup.end());
    return bottomup;
}

struct CbspCatalog
{
    std::vector<polymba::PolyMBAFamily> fams32;
    std::vector<polymba::PolyMBAFamily> fams64;
};

static CbspCatalog BuildCatalog(uint64_t seed)
{
    CbspCatalog cat;
    polymba::BuildOptions opt;
    opt.Count = CatalogCount;
    opt.InverseCapD = CatalogInverseCapD;
    opt.VerifyBits = CatalogVerifyBits;
    opt.VerifyRandom = CatalogVerifyRandom;
    opt.A2Valuation = CatalogA2Valuation;
    opt.MaxTriesMult = CatalogMaxTriesMult;
    opt.Seed = seed;
    opt.Width = 32;
    cat.fams32 = polymba::BuildFamilies(opt);
    // XOR'ing the 32bit catalog seed with the golden ratio constant for the 64bit catalog keeps the two family sets independent even on an identical caller seed, so a function's 32bit and 64bit site picks don't correlate
    opt.Seed = seed ^ 0x9E3779B97F4A7C15ULL;
    opt.Width = 64;
    cat.fams64 = polymba::BuildFamilies(opt);
    return cat;
}

static Value* HornerEmit(IRBuilder<>& b, const polymba::Poly& p, Value* x, IntegerType* vt, bool tagadds)
{
    int deg = -1;
    for (int k = (int)p.C.size() - 1; k >= 0; --k)
        if (p.C[k]) { deg = k; break; }
    if (deg < 0) return ConstantInt::get(vt, 0);
    if (deg == 0) return ConstantInt::get(vt, p.C[0]);
    Value* acc = ConstantInt::get(vt, p.C[deg]);
    for (int k = deg - 1; k >= 0; --k)
    {
        acc = b.CreateMul(acc, x, "cbsp.h.mul");
        acc = b.CreateAdd(acc, ConstantInt::get(vt, p.C[k]), "cbsp.h.add");
        if (tagadds)
            if (auto* ii = dyn_cast<Instruction>(acc)) TagMbagenEscalate(ii);
    }
    return acc;
}

static std::vector<Value*> HarvestAuxPool(Function& f)
{
    std::vector<Value*> pool;
    pool.reserve(AuxPoolCap);
    for (Argument& a : f.args())
    {
        if (a.getType()->isIntegerTy())
        {
            pool.push_back(&a);
            if (pool.size() >= AuxPoolCap) return pool;
        }
    }
    for (BasicBlock& bb : f)
    {
        for (Instruction& i : bb)
        {
            if (!i.getType()->isIntegerTy()) continue;
            if (isa<AllocaInst>(&i)) continue;
            if (isa<CallInst>(&i)) continue;
            if (isa<InvokeInst>(&i)) continue;
            if (auto* ld = dyn_cast<LoadInst>(&i))
                if (ld->isVolatile()) continue;
            pool.push_back(&i);
            if (pool.size() >= AuxPoolCap) return pool;
        }
    }
    return pool;
}

static bool ValueDominatesBB(const DominatorTree& dt, Value* v, BasicBlock* bb, Function* f)
{
    if (auto* a = dyn_cast<Argument>(v)) return a->getParent() == f;
    if (auto* i = dyn_cast<Instruction>(v))
    {
        BasicBlock* defbb = i->getParent();
        if (defbb == bb) return false;
        return dt.dominates(defbb, bb);
    }
    return false;
}

static Value* PickAuxForBB(const std::vector<Value*>& pool, const DominatorTree& dt, BasicBlock* bb, Function* f, std::mt19937_64& rng)
{
    if (pool.empty()) return nullptr;
    std::vector<Value*> ok;
    ok.reserve(pool.size());
    for (Value* v : pool)
        if (ValueDominatesBB(dt, v, bb, f)) ok.push_back(v);
    if (ok.empty()) return nullptr;
    return ok[(size_t)(rng() % ok.size())];
}

static Value* CastAuxToWidth(IRBuilder<>& b, Value* aux, IntegerType* vt)
{
    auto* src = dyn_cast<IntegerType>(aux->getType());
    if (!src) return nullptr;
    if (src->getBitWidth() == vt->getBitWidth()) return aux;
    if (src->getBitWidth() < vt->getBitWidth()) return b.CreateZExt(aux, vt, "cbsp.aux.ze");
    return b.CreateTrunc(aux, vt, "cbsp.aux.tr");
}

static Value* EmitAuxWrappedXorConst(IRBuilder<>& b, Value* vin, uint64_t delta, unsigned bw, Value* auxcast, IntegerType* vt, std::mt19937_64& rng)
{
    uint64_t da = MaskW(rng(), bw);
    uint64_t db = MaskW(delta ^ da, bw);
    Value* v = vin;
    v = b.CreateXor(v, ConstantInt::get(vt, da), "cbsp.aux.d1");
    v = InsertFoldBarrier(b, v);
    v = b.CreateXor(v, auxcast, "cbsp.aux.a1");
    v = InsertFoldBarrier(b, v);
    v = b.CreateXor(v, ConstantInt::get(vt, db), "cbsp.aux.d2");
    v = InsertFoldBarrier(b, v);
    v = b.CreateXor(v, auxcast, "cbsp.aux.a2");
    return v;
}

static Value* EmitAuxWrappedXorValue(IRBuilder<>& b, Value* vin, Value* deltart, unsigned bw, Value* auxcast, IntegerType* vt, std::mt19937_64& rng)
{
    uint64_t da = MaskW(rng(), bw);
    Value* dac = ConstantInt::get(vt, da);
    Value* dbrt = b.CreateXor(deltart, dac, "cbsp.phi.db");
    Value* v = vin;
    v = b.CreateXor(v, dac, "cbsp.aux.d1");
    v = InsertFoldBarrier(b, v);
    v = b.CreateXor(v, auxcast, "cbsp.aux.a1");
    v = InsertFoldBarrier(b, v);
    v = b.CreateXor(v, dbrt, "cbsp.aux.d2");
    v = InsertFoldBarrier(b, v);
    v = b.CreateXor(v, auxcast, "cbsp.aux.a2");
    return v;
}

struct MorphPlan
{
    BasicBlock* bb = nullptr;
    bool isphi = false;
    uint64_t delta = 0;
    std::vector<std::pair<BasicBlock*, uint64_t>> phideltas;
};

static std::vector<MorphPlan> BuildMorphPlans(const std::vector<BasicBlock*>& chain, unsigned bw, std::mt19937_64& rng, uint64_t& outdeltasumfixed, bool& outusestracker)
{
    std::vector<MorphPlan> plans;
    plans.reserve(chain.size());
    outdeltasumfixed = 0;
    outusestracker = false;
    for (BasicBlock* bb : chain)
    {
        MorphPlan mp;
        mp.bb = bb;
        bool wantphi = PhiEnable;
        if (wantphi)
        {
            SmallPtrSet<BasicBlock*, 8> uniq;
            for (BasicBlock* p : predecessors(bb)) uniq.insert(p);
            if (uniq.size() < PhiMinPreds || uniq.empty())
            {
                wantphi = false;
            }
            else
            {
                mp.isphi = true;
                mp.phideltas.reserve(uniq.size());
                for (BasicBlock* p : uniq)
                {
                    uint64_t d = MaskW(rng(), bw);
                    mp.phideltas.emplace_back(p, d);
                }
                outusestracker = true;
            }
        }
        if (!wantphi)
        {
            mp.isphi = false;
            mp.delta = MaskW(rng(), bw);
            // fixed deltas XOR reduce to a single compile time constant we subtract at use.BB; PHI deltas can't fold this way (they're perpred), so they go through the runtime tracker alloca instead
            outdeltasumfixed ^= mp.delta;
        }
        plans.push_back(std::move(mp));
    }
    return plans;
}

static bool EmitProd(Function& f, const SitePick& pick, const DominatorTree& dt, const LoopInfo& li, const CbspCatalog& catalog, const std::vector<Value*>& auxpool)
{
    LLVMContext& ctx = f.getContext();
    IntegerType* vt = IntegerType::get(ctx, pick.bw);
    BasicBlock& entry = f.getEntryBlock();
    BasicBlock* usebb = pick.user->getParent();
    auto& rng = CbspRng();
    auto barr = [&](IRBuilder<>& b, Value* v) -> Value* { return InsertFoldBarrier(b, v); };

    const bool escalated = pick.escalated;
    (void)escalated;

    std::vector<BasicBlock*> chain = CollectChain(dt, li, usebb, &entry);

    uint64_t deltasumfixed = 0;
    bool usestracker = false;
    std::vector<MorphPlan> plans = BuildMorphPlans(chain, pick.bw, rng, deltasumfixed, usestracker);

    const auto& fams = (pick.bw == 32) ? catalog.fams32 : catalog.fams64;
    const bool havepoly = !fams.empty();
    const uint64_t k = pick.k->getZExtValue();

    uint64_t enc0;
    uint64_t decodedeltaconst = 0;
    size_t famidx = 0;
    if (havepoly)
    {
        famidx = (size_t)(rng() % fams.size());
        enc0 = MaskW(fams[famidx].F.Eval(k, pick.bw), pick.bw);
    }
    else
    {
        // fallback path when the catalog built empty for this width; XOR only encoding stays correct, just loses the Horner outer layer
        uint64_t extra = MaskW(rng(), pick.bw);
        enc0 = MaskW(k ^ deltasumfixed ^ extra, pick.bw);
        decodedeltaconst = extra;
    }

    IRBuilder<> allocab(&entry, entry.getFirstNonPHIIt());
    // name "cbsp.slot" is downstream contract; identifier passes and other MBA stages grep for it to recognise CBSP owned storage
    AllocaInst* slot = allocab.CreateAlloca(vt, nullptr, "cbsp.slot");

    AllocaInst* tracker = nullptr;
    if (usestracker)
    {
        // name "cbsp.tracker" is downstream contract paired with "cbsp.slot"
        tracker = allocab.CreateAlloca(vt, nullptr, "cbsp.tracker");
    }

    Instruction* afteralloca = (tracker ? tracker : slot)->getNextNode();
    if (!afteralloca)
    {
        if (tracker) tracker->eraseFromParent();
        slot->eraseFromParent();
        return false;
    }
    IRBuilder<> storeb(afteralloca);

    StoreInst* st0 = storeb.CreateStore(ConstantInt::get(vt, enc0), slot);
    st0->setVolatile(true);
    if (tracker)
    {
        StoreInst* tst0 = storeb.CreateStore(ConstantInt::get(vt, 0), tracker);
        tst0->setVolatile(true);
    }

    for (const MorphPlan& mp : plans)
    {
        BasicBlock* bb = mp.bb;
        PHINode* deltaphi = nullptr;
        if (mp.isphi)
        {
            IRBuilder<> phib(bb, bb->getFirstNonPHIIt());
            deltaphi = phib.CreatePHI(vt, (unsigned)mp.phideltas.size(), "cbsp.phi.delta");
            for (const auto& pd : mp.phideltas)
                deltaphi->addIncoming(ConstantInt::get(vt, pd.second), pd.first);
        }
        IRBuilder<> mb(bb, bb->getFirstNonPHIIt());
        LoadInst* ld = mb.CreateLoad(vt, slot, /*isVolatile=*/true, "cbsp.m.load");
        Value* ldb = barr(mb, ld);

        Value* xored = nullptr;
        if (mp.isphi)
        {
            Value* auxraw = PickAuxForBB(auxpool, dt, bb, &f, rng);
            if (auxraw)
            {
                Value* auxcast = CastAuxToWidth(mb, auxraw, vt);
                xored = EmitAuxWrappedXorValue(mb, ldb, deltaphi, pick.bw, auxcast, vt, rng);
            }
            if (!xored) xored = mb.CreateXor(ldb, deltaphi, "cbsp.m.phi.xor");
        }
        else
        {
            Value* auxraw = PickAuxForBB(auxpool, dt, bb, &f, rng);
            if (auxraw)
            {
                Value* auxcast = CastAuxToWidth(mb, auxraw, vt);
                xored = EmitAuxWrappedXorConst(mb, ldb, mp.delta, pick.bw, auxcast, vt, rng);
            }
            if (!xored) xored = mb.CreateXor(ldb, ConstantInt::get(vt, mp.delta), "cbsp.m.xor");
        }

        Value* xoredb = barr(mb, xored);
        StoreInst* mst = mb.CreateStore(xoredb, slot);
        mst->setVolatile(true);

        if (mp.isphi)
        {
            LoadInst* tld = mb.CreateLoad(vt, tracker, /*isVolatile=*/true, "cbsp.t.load");
            Value* txored = mb.CreateXor(tld, deltaphi, "cbsp.t.phi.xor");
            StoreInst* tst = mb.CreateStore(txored, tracker);
            tst->setVolatile(true);
        }
    }

    IRBuilder<> decb(pick.user);
    LoadInst* ldfinal = decb.CreateLoad(vt, slot, /*isVolatile=*/true, "cbsp.load");
    Value* ldb = barr(decb, ldfinal);

    Value* unwrapdeltavalue = nullptr;
    uint64_t unwrapdeltaconstant = 0;
    bool unwrapisruntime = false;

    if (usestracker)
    {
        LoadInst* ldtracker = decb.CreateLoad(vt, tracker, /*isVolatile=*/true, "cbsp.t.dec.load");
        Value* tb = barr(decb, ldtracker);
        Value* combo = decb.CreateXor(tb, ConstantInt::get(vt, deltasumfixed), "cbsp.t.combo");
        unwrapdeltavalue = barr(decb, combo);
        unwrapisruntime = true;
    }
    else
    {
        unwrapdeltaconstant = deltasumfixed;
        unwrapisruntime = false;
    }

    Value* decoded;
    if (havepoly)
    {
        Value* unwrapped = nullptr;
        Value* auxraw = PickAuxForBB(auxpool, dt, usebb, &f, rng);
        if (auxraw)
        {
            Value* auxcast = CastAuxToWidth(decb, auxraw, vt);
            if (unwrapisruntime)
                unwrapped = EmitAuxWrappedXorValue(decb, ldb, unwrapdeltavalue, pick.bw, auxcast, vt, rng);
            else
                unwrapped = EmitAuxWrappedXorConst(decb, ldb, unwrapdeltaconstant, pick.bw, auxcast, vt, rng);
        }
        if (!unwrapped)
        {
            if (unwrapisruntime) unwrapped = decb.CreateXor(ldb, unwrapdeltavalue, "cbsp.unwrap");
            else unwrapped = decb.CreateXor(ldb, ConstantInt::get(vt, unwrapdeltaconstant), "cbsp.unwrap");
        }
        Value* unwb = barr(decb, unwrapped);
        Value* h = HornerEmit(decb, fams[famidx].FInv, unwb, vt, TagHornerEmits);
        decoded = barr(decb, h);
    }
    else
    {
        Value* rawdec = nullptr;
        if (unwrapisruntime)
        {
            Value* extramixed = decb.CreateXor(unwrapdeltavalue, ConstantInt::get(vt, decodedeltaconst), "cbsp.fb.mix");
            rawdec = decb.CreateXor(ldb, extramixed, "cbsp.dec");
        }
        else
        {
            uint64_t fallbackconstpart = MaskW(deltasumfixed ^ decodedeltaconst, pick.bw);
            Value* auxraw = PickAuxForBB(auxpool, dt, usebb, &f, rng);
            if (auxraw)
            {
                Value* auxcast = CastAuxToWidth(decb, auxraw, vt);
                rawdec = EmitAuxWrappedXorConst(decb, ldb, fallbackconstpart, pick.bw, auxcast, vt, rng);
            }
            if (!rawdec)
                rawdec = decb.CreateXor(ldb, ConstantInt::get(vt, fallbackconstpart), "cbsp.dec");
        }
        decoded = barr(decb, rawdec);
    }

    pick.user->setOperand(pick.opidx, decoded);
    return true;
}

static uint64_t ArxCt(uint64_t x, uint64_t m, uint64_t c, unsigned r)
{
    uint64_t t = x ^ m;
    t = (t << r) | (t >> (64 - r));
    return t + c;
}

static std::pair<uint64_t, uint64_t> ArxPickMC(uint64_t xin, uint64_t xout, unsigned r, std::mt19937_64& rng)
{
    uint64_t m = rng();
    uint64_t t = xin ^ m;
    uint64_t rotated = (t << r) | (t >> (64 - r));
    uint64_t c = xout - rotated;
    (void)ArxCt;
    return {m, c};
}

static void EmitArxMix(IRBuilder<>& b, AllocaInst* slot, uint64_t m, uint64_t c, unsigned r, Module* mod)
{
    LLVMContext& ctx = b.getContext();
    Type* i64 = Type::getInt64Ty(ctx);
    LoadInst* ld = b.CreateLoad(i64, slot, "arx.acc");
    ld->setVolatile(true);
    Value* x = b.CreateXor(ld, ConstantInt::get(i64, m), "arx.x");
    Function* fshl = Intrinsic::getDeclaration(mod, Intrinsic::fshl, i64);
    Value* rot = b.CreateCall(fshl, {x, x, ConstantInt::get(i64, (uint64_t)r)}, "arx.r");
    Value* added = b.CreateAdd(rot, ConstantInt::get(i64, c), "arx.c");
    StoreInst* st = b.CreateStore(added, slot);
    st->setVolatile(true);
}

static bool BBContainsEHPad(BasicBlock& bb)
{
    for (Instruction& i : bb)
        if (i.isEHPad()) return true;
    return false;
}

static bool IsArxConvertibleTerminator(Instruction* term)
{
    return term && isa<BranchInst>(term);
}

static bool EdgeIsArxSplittable(BasicBlock* p, BasicBlock* b)
{
    if (BBContainsEHPad(*p)) return false;
    if (BBContainsEHPad(*b)) return false;
    return true;
}

static bool RunArxBranchOnFunction(Function& f, std::mt19937_64& rng)
{
    LLVMContext& ctx = f.getContext();
    Module* m = f.getParent();
    Type* i64 = Type::getInt64Ty(ctx);
    PointerType* ptrT = PointerType::get(ctx, 0);
    const unsigned r = ArxRot;
    // ArxRot comes from a constexpr set to 13 (prime, non-trivial), but a future refactor could parameterize it; guard anyway so a bad config can't produce an undef shift
    if (r == 0 || r >= 64) return false;

    std::vector<BasicBlock*> originalbbs;
    originalbbs.reserve(f.size());
    for (BasicBlock& bb : f) originalbbs.push_back(&bb);
    if (originalbbs.empty()) return false;

    BasicBlock& entry = f.getEntryBlock();

    DenseMap<BasicBlock*, uint64_t> expected;
    expected.reserve(originalbbs.size());
    uint64_t seedfn = rng();
    for (BasicBlock* bb : originalbbs)
    {
        if (bb == &entry) expected[bb] = seedfn;
        else expected[bb] = rng();
    }

    IRBuilder<> allocab(&entry, entry.getFirstNonPHIIt());
    // name "arx.slot" is downstream contract; identifier passes grep for it to recognise ARX owned storage
    AllocaInst* arxslot = allocab.CreateAlloca(i64, nullptr, "arx.slot");
    Instruction* afteralloca = arxslot->getNextNode();
    if (!afteralloca)
    {
        arxslot->eraseFromParent();
        return false;
    }
    IRBuilder<> seedb(afteralloca);
    StoreInst* seedst = seedb.CreateStore(ConstantInt::get(i64, seedfn), arxslot);
    seedst->setVolatile(true);

    struct EdgeRecord { BasicBlock* p; BasicBlock* b; unsigned succidx; };
    std::vector<EdgeRecord> edges;
    for (BasicBlock* p : originalbbs)
    {
        Instruction* term = p->getTerminator();
        if (!term) continue;
        if (BBContainsEHPad(*p)) continue;
        for (unsigned i = 0, n = term->getNumSuccessors(); i < n; ++i)
        {
            BasicBlock* b = term->getSuccessor(i);
            if (!EdgeIsArxSplittable(p, b)) continue;
            edges.push_back({p, b, i});
        }
    }

    size_t edgessplit = 0;
    size_t mixesemitted = 0;
    for (const EdgeRecord& e : edges)
    {
        BasicBlock* edgebb = SplitEdge(e.p, e.b, nullptr, nullptr, nullptr, "arx.edge");
        if (!edgebb) continue;
        ++edgessplit;
        auto [m2, c2] = ArxPickMC(expected[e.p], expected[e.b], r, rng);
        IRBuilder<> mixb(edgebb, edgebb->getFirstNonPHIIt());
        EmitArxMix(mixb, arxslot, m2, c2, r, m);
        ++mixesemitted;
    }
    (void)edgessplit;

    unsigned addrtablecounter = 0;
    size_t branchesindir = 0;
    for (BasicBlock* b : originalbbs)
    {
        if (BBContainsEHPad(*b)) continue;
        Instruction* term = b->getTerminator();
        if (!term) continue;
        // SwitchInst stays as is; the jump table in a switch is more complex to rewrite as indirectbr with-decode and gets handled by a separate pass; invokes/returns/resumes aren't convertible either
        if (isa<SwitchInst>(term)) continue;
        if (!IsArxConvertibleTerminator(term)) continue;

        auto* br = cast<BranchInst>(term);

        SmallVector<BasicBlock*, 4> succs;
        for (unsigned i = 0, n = br->getNumSuccessors(); i < n; ++i)
            succs.push_back(br->getSuccessor(i));
        if (succs.empty()) continue;

        // entries are encoded as ADD rather than XOR because PE/COFF emits a standard base+offset relocation on a BlockAddress + integer, while the backend rejects XOR of a BlockAddress as a static initializer; runtime decode does the matching SUB
        SmallVector<Constant*, 4> entries(succs.size());
        uint64_t expB = expected[b];
        for (unsigned i = 0; i < succs.size(); ++i)
        {
            Constant* ba = BlockAddress::get(&f, succs[i]);
            Constant* baInt = ConstantExpr::getPtrToInt(ba, i64);
            Constant* key = ConstantInt::get(i64, expB);
            entries[i] = ConstantExpr::getAdd(baInt, key);
        }
        ArrayType* at = ArrayType::get(i64, succs.size());
        // table name "__cbsp_arx_tbl_" + funcname + counter is downstream contract; linker stripper tooling finds and relocates these per-function
        std::string tblname = "__cbsp_arx_tbl_" + f.getName().str() + "_" + std::to_string(addrtablecounter++);
        auto* tbl = new GlobalVariable(*m, at, /*isConstant=*/false,
            GlobalValue::PrivateLinkage, ConstantArray::get(at, entries), tblname);
        tbl->setUnnamedAddr(GlobalValue::UnnamedAddr::Global);

        IRBuilder<> rb(br);
        Type* i32 = Type::getInt32Ty(ctx);
        Constant* zero32 = ConstantInt::get(i32, 0);
        Value* idx = nullptr;
        if (br->isConditional())
        {
            Constant* idxT = ConstantInt::get(i32, 0);
            Constant* idxF = ConstantInt::get(i32, 1);
            idx = rb.CreateSelect(br->getCondition(), idxT, idxF, "arx.br.idx");
        }
        else
        {
            idx = ConstantInt::get(i32, 0);
        }
        Value* gep = rb.CreateGEP(at, tbl, {zero32, idx}, "arx.br.gep");
        Value* ent = rb.CreateLoad(i64, gep, "arx.br.ent");
        LoadInst* acc = rb.CreateLoad(i64, arxslot, "arx.br.acc");
        acc->setVolatile(true);
        // table entry was ptrtoint(succ) + expected[B]; when acc == expected[B] the SUB recovers ptrtoint(succ) exactly, so an attacker who skipped any CFG edge mix lands on a bogus address at indirectbr
        Value* dec = rb.CreateSub(ent, acc, "arx.br.dec");
        Value* tgt = rb.CreateIntToPtr(dec, ptrT, "arx.br.tgt");

        auto* ibr = IndirectBrInst::Create(tgt, succs.size(), br);
        for (BasicBlock* sc : succs) ibr->addDestination(sc);
        ibr->copyMetadata(*br);
        br->eraseFromParent();
        ++branchesindir;
    }

    return branchesindir > 0 || mixesemitted > 0;
}

bool RunCrossBlockSubstrate(Module& m)
{
    if (!CBSPEnable) return false;

    CbspCatalog catalog;
    const uint64_t catseed = CbspRng()();
    catalog = BuildCatalog(catseed);

    bool mutated = false;
    for (Function& f : m)
    {
        if (f.isDeclaration()) continue;
        if (!HasObfuscateAnnotation(f)) continue;
        if (HasNoObfuscateAnnotation(f)) continue;

        DominatorTree dt;
        dt.recalculate(f);
        LoopInfo li;
        li.analyze(dt);

        std::vector<Value*> auxpool = HarvestAuxPool(f);

        SitePick pick = FindFirstSite(f, li);
        if (pick.user)
        {
            if (EmitProd(f, pick, dt, li, catalog, auxpool)) mutated = true;
        }

       
        if (ArxBranchEnable)
        {
            if (RunArxBranchOnFunction(f, CbspArxRng())) mutated = true;
        }
    }

    return mutated;
}

PreservedAnalyses CrossBlockSubstratePass::run(Module& m, ModuleAnalysisManager&)
{
    bool mutated = RunCrossBlockSubstrate(m);
    return mutated ? PreservedAnalyses::none() : PreservedAnalyses::all();
}