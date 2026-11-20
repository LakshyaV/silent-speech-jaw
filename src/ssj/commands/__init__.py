"""Sub-commands of the ``ssj`` CLI.

A command is a module in this package exposing::

    NAME = "train"                       # sub-command name
    HELP = "one-line summary"
    ORDER = 50                           # optional sort key for --help
    def configure(parser): ...           # add arguments
    def run(args) -> int | None: ...     # do the work

Adding a file here is all it takes to register a command.
"""
