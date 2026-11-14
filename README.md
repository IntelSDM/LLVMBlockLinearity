# Block Linearity
Built for LLVM 19.1 - Windows x64 <br> <br>
* This pass essentially takes constants that are used in functions, puts them into the stack encrypted at the first block.
* Then horner encodes them through each block(Each overhead block changes the encode, so you must track the horner algos).
* Decodes them at usage.
* It also flattens control flow with basic ARX indirect jumps(the keys are stored upwards so you only know them at site, you must follow control flow linearly to get them) so you can't traverse all blocks at once.


This creates a system of localised homomorphism, the idea is that once you actually apply proper control flow obfuscation, the indirects will not be so easy to traverse.<br>
Therefore; once lifted you must apply your deobfuscation passes much slower, on each individual block or the lifter itself needs to fully defeat your control flow symbolically to optimise the entire function. <br>
Additionally, it completely prevents localised block analysis, so if you can't reverse engineer 1 mechanism above head as it broke your tooling, then you can't analyse anything below it. <br>
If you were lets say trying to only deobfuscate 1 thing for example [augur](https://plugins.hex-rays.com/0xdea/augur/augur) which only wants to go after strings, it can't, as everything downstream needs full deobfuscation using overhead values. <br>
<br>
<img width="323" height="768" alt="image" src="https://github.com/user-attachments/assets/766dba80-8115-4ae3-aed4-4a5a08770ada" /> <br>
<img width="595" height="479" alt="image" src="https://github.com/user-attachments/assets/97915d96-1c6a-4f4e-8f4e-ace1de304399" /> <br>
<img width="298" height="927" alt="image" src="https://github.com/user-attachments/assets/fdf8a15f-78b3-49d9-9ef7-00e9cb76771b" /> <br>
<img width="453" height="857" alt="image" src="https://github.com/user-attachments/assets/21d411d0-e3da-4cd0-9709-d84bf22f667d" /> <br>



## How To Build
* Download [LLVM 19.1](https://github.com/llvm/llvm-project/releases/tag/llvmorg-19.1.0)
* Download [VCPKG](https://github.com/microsoft/vcpkg)
* VCPKG install asmjit
* Open build.bat in a text editor.
* Find the LLVM 19.1 references in the command line, Change them to your file path.
* Find the VCPKG references in the command line, Change them to your file path.
* Run the build script
* Build your application with the ``` -load-pass-plugin ``` Param to the FunctionPass.dll
