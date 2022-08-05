let struct = (rg -no 'pub (?:struct|enum) (Checked([^\s])+)' ../src/typechecker.rs -r '$1' | lines | split column ':' start_line struct)


