{
    mkShell,
    python3,
    pyright,
    perf,
    strace,
    py-spy,
}:
mkShell {
    buildInputs = [
        (python3.withPackages (p: with p; [
            networkx
            z3-solver
            ortools
            pytest
        ]))
    ];
    nativeBuildInputs = [
        pyright
        perf
        strace
        py-spy
    ];

    shellHook = ''
      export PYTHONPATH="$PYTHONPATH:$PWD/src"
    '';
}
