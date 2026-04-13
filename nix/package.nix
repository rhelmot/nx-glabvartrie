{
    lib,
    buildPythonPackage,
    setuptools,
    networkx,
    z3-solver,
}:
buildPythonPackage {
    pname = "nx-glabtrie";
    version = "0.1.0";
    src = ./..;
    pyproject = true;

    build-system = [ setuptools ];
    dependencies = [ networkx z3-solver ];

    meta = {
        license = with lib.licenses; [ publicDomain ];
    };
}
