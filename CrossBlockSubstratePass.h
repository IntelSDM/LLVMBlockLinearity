#pragma once

#include "llvm/IR/PassManager.h"
#include "llvm/Passes/PassPlugin.h"
#include "llvm/Support/CommandLine.h"

struct CrossBlockSubstratePass : llvm::PassInfoMixin<CrossBlockSubstratePass>
{
    llvm::PreservedAnalyses run(llvm::Module& m, llvm::ModuleAnalysisManager& mam);
};

bool RunCrossBlockSubstrate(llvm::Module& m);

extern llvm::cl::opt<bool> CBSPEnable;