# License

Copyright (c) 2026 redcobra1652. All rights reserved.

This repository, and everything in it — including but not limited to the
engine source code (`nnue_engine.cpp` and every other `.cpp`/`.h` file),
the training and data pipeline (`train.py`, `model.py`, `serialize.py`,
`data.py`, `convert.py`, `convert_lichess.py`, and related scripts), trained
network weights and checkpoints (`checkpoints/`), and all documentation
(including `CLAUDE.md`) — is the sole, original work of the author and is
**not** open source. This project is provided for others to use and observe,
not to copy, build on, or redistribute.

## What is permitted

You may **use** this software as an end user: run the compiled engine,
play against it (including via any bot instance of it the author hosts,
e.g. on Lichess), and observe its output, for your own personal,
non-commercial purposes.

## What is not permitted

Except as explicitly stated above, no permission is granted to:

- Copy, reproduce, fork, modify, translate, or create derivative works
  from any source code, model weights, training data pipeline, or
  documentation in this repository, in whole or in part.
- Use any portion of this repository's source code, architecture, trained
  weights, or documentation in another project, publication, product, or
  codebase — commercial or non-commercial.
- Sublicense, sell, rent, or otherwise transfer any rights to this
  repository or its contents to any third party.
- Claim authorship, in whole or in part, of any of this repository's
  contents.

## Distribution

This repository may only be obtained through the author's official GitHub
repository (**https://github.com/redcobra1652/ChessEnginev2**) or directly
from the author. **Redistribution through any other source** — including
but not limited to mirrors, forks intended for redistribution, third-party
hosting, package registries, or other code-sharing platforms — **is
prohibited.**

## Reservation of rights

All rights not expressly granted above are reserved by the author. Nothing
in this license shall be construed as granting any license or right under
any patent, trademark, or other intellectual property right of the author,
by implication, estoppel, or otherwise.

## No warranty

This software is provided "as is", without warranty of any kind, express
or implied, including but not limited to the warranties of merchantability,
fitness for a particular purpose, and noninfringement. In no event shall
the author be liable for any claim, damages, or other liability arising
from, out of, or in connection with the software or the use or other
dealings in the software.

## Third-party components

This repository's own restrictive terms above apply only to the author's
original work. Where this project depends on or is used alongside
separately-licensed third-party software — for example
[Stockfish](https://github.com/official-stockfish/Stockfish) (GPLv3, used
locally as a benchmarking opponent, not vendored in this repository — see
`README.md`), [fastchess](https://github.com/Disservin/fastchess) (vendored
locally in `fastchess_src/`, gitignored), or
[lichess-bot](https://github.com/lichess-bot-devs/lichess-bot) (cloned
locally by `setup_lichess_bot.sh` into `lichess-bot/`, gitignored) — none of
which are part of this repository — those components remain
governed by their own respective licenses, not this one.
