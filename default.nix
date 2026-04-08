let
	deps = import ./nix/tamal {};
	pkgs = import deps.nixpkgs {};
	pkg = pkgs.callPackage ./nix/package.nix {};
in pkg
