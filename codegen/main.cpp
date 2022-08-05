#include <fstream>
#include <iostream>

// External includes
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#include "nlohmann_json/single_include/nlohmann/json.hpp"

#include "llvm/ADT/APFloat.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/BasicBlock.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DerivedTypes.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/Type.h"
#include "llvm/IR/Verifier.h"
#pragma clang diagnostic pop

using json = nlohmann::json;

//// Owns other LLVM objects.
// static llvm::LLVMContext TheContext;
//// Keeps track of the current place to insert instructions
// static llvm::IRBuilder<> Builder(TheContext);
//// Module contains top-level functions and global variables.
// static std::unique_ptr<llvm::Module> TheModule;
//// Keeps track of which names are defined in the current scope and what their
//// llvm representation is
// static std::map<std::string, llvm::Value *> NamedValues;
//

int main() {
    std::ifstream f{"../ast.json"};

    auto json = json::parse(f);
}
