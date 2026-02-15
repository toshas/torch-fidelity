# CLAUDE.md

## Code editing preferences

- Minimize git diff: when fixing a bug, prefer swapping labels/names over reordering code blocks.
- Do not rearrange code sections unnecessarily; keep the original structure intact.
- Do not change computations that are mathematically equivalent (e.g., swapping `A = f(x,y); B = A.T` to `B = f(y,x); A = B.T`) when only the downstream labels need fixing.
- Prefer the smallest possible change that achieves correctness.
