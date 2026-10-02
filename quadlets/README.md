<!-- SPDX-License-Identifier: GPL-3.0-only -->
<!-- Copyright (C) 2026 Mateusz Okulanis -->
# Example Quadlet units

`install.sh quadlets` writes the real units into
`~/.config/containers/systemd/`, filled in from the plan: the paths
carry the installing user's home directory, and `ngl`, `ctx` and the
draft model are decided by `install.sh plan` from the machine it runs
on. These three files are what it produced on the reference machine,
kept here so the shape can be read without installing anything and so
a change to the generator shows up as a diff.

Do not copy them into place. The ports, the subnet and the layer
counts are this machine's; `./install.sh detect plan --dry-run` prints
what yours would be.
