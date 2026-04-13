let
	deps = import ./nix/tamal {};
	pkgs = import deps.nixpkgs {};
	pkg = pkgs.python3.pkgs.callPackage ./nix/package.nix {};
in pkg
