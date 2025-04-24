rm .\build
mkdir .\build
cd .\build
cmake .. -G "Visual Studio 16 2019"
cmake --build . --config Release
cd ..
copy .\build\Release\* .\Release\