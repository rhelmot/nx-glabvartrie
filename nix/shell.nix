{
    mkShell,
    python3,
    pyright,
}:
mkShell {
    buildInputs = [
        (python3.withPackages (p: with p; [
            networkx
            pytest
        ]))
    ];
    nativeBuildInputs = [
        pyright
    ];
}
