{
    lib,
    python3,
}:
python3.pkgs.buildPythonPackage {
    pname = "nx-glabtrie";
    version = "0.1.0";
    src = ./..;
    pyproject = true;

    build-system = with python3.pkgs; [ setuptools ];
    dependencies = with python3.pkgs; [ networkx ];

    meta = {
        license = with lib.licenses; [ publicDomain ];
    };
}
