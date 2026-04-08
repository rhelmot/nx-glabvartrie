let
	deps = import ./nix/tamal {};
	pkgs = import deps.nixpkgs {};
	shell = pkgs.callPackage ./nix/shell.nix {};
in shell
