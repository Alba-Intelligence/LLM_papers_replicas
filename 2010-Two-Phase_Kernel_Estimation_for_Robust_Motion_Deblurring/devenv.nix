{
  pkgs,
  lib,
  config,
  inputs,
  ...
}:

let
  llmsPkgs = inputs.llms.packages.${pkgs.stdenv.hostPlatform.system};
in
{
  # https://devenv.sh/basics/
  env.GREET = "devenv";

  # https://devenv.sh/packages/
  packages =
    (with pkgs; [
      git

      # Utilities
      curl
      ncurses
      openspecfun
      openssl
      unzip
      util-linux

      which
      zlib

      # Building
      patch
      autoconf
      binutils
      clang
      cmake
      expat
      gcc
      gfortran
      gmp
      gnumake
      gperf
      libxml2
      m4
      nss
      stdenv.cc

      # DB
      # sqlite

      # IDE
      # code-cursor
      # amp-cli
      bash # Required by Speckit scripts
      vscode

    ])
    ++ (with llmsPkgs; [
      cursor-agent
      gemini-cli # Broken build
      kilocode-cli
      opencode
      openskills
      openspec
      spec-kit
    ]);

  # https://devenv.sh/languages/
  languages = {
    julia.enable = true;

    python = {
      enable = true;
      lsp.enable = true;
      uv.enable = true;
    };
  };

  # https://devenv.sh/processes/
  # processes.dev.exec = "${lib.getExe pkgs.watchexec} -n -- ls -la";

  # https://devenv.sh/services/
  # services.postgres.enable = true;

  # https://devenv.sh/scripts/
  scripts.hello.exec = ''
    echo hello from $GREET
  '';

  # https://devenv.sh/basics/
  enterShell = ''
    export LD_LIBRARY_PATH=${pkgs.openspecfun}/lib:${pkgs.zlib}/lib::${pkgs.curl}/lib:$LD_LIBRARY_PATH

    hello         # Run scripts directly
    git --version # Use packages
    echo
  '';

  # https://devenv.sh/tasks/
  # tasks = {
  #   "myproj:setup".exec = "mytool build";
  #   "devenv:enterShell".after = [ "myproj:setup" ];
  # };

  # https://devenv.sh/tests/
  enterTest = ''
    echo "Running tests"
    git --version | grep --color=auto "${pkgs.git.version}"
  '';

  # https://devenv.sh/git-hooks/
  # git-hooks.hooks.shellcheck.enable = true;

  # See full reference at https://devenv.sh/reference/options/
}
