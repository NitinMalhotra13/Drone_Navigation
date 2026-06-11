"""fix_unicode.py — strips non-cp1252 characters from all src files"""
import os

files = [
    'src/generate_static_obstacles.py',
    'src/multi_drone_coverage_env.py',
    'src/fa_coverage.py',
    'src/ra_coverage.py',
    'src/visualize_multi_drone.py',
    'src/run_integrated.py',
    'src/train_multi_drone_ppo.py',
    'src/dynamic_obstacles.py',
    'src/generate_terrain.py',
]

REPLACEMENTS = [
    ('\u2714', '[OK]'),
    ('\u2192', '->'),
    ('\u2190', '<-'),
    ('\u2191', '^'),
    ('\u2193', 'v'),
    ('\u00d7', 'x'),
    ('\u221d', 'prop to'),
    ('\u00b2', '2'),
    ('\u2248', '~='),
    ('\u221e', 'inf'),
    ('\u03b3', 'gamma'),
    ('\u03b1', 'alpha'),
    ('\u03b2', 'beta'),
    ('\u2260', '!='),
    ('\u2265', '>='),
    ('\u2264', '<='),
    ('\u00b7', '*'),
    ('\u2022', '-'),
    ('\u2500', '-'),
    ('\u2502', '|'),
    ('\u2550', '='),
    ('\u2551', '|'),
    ('\u2554', '+'),
    ('\u2557', '+'),
    ('\u255a', '+'),
    ('\u255d', '+'),
    ('\u256c', '+'),
    ('\u250c', '+'),
    ('\u2510', '+'),
    ('\u2514', '+'),
    ('\u2518', '+'),
    ('\u251c', '+'),
    ('\u2524', '+'),
    ('\u252c', '+'),
    ('\u2534', '+'),
    ('\u253c', '+'),
    ('\u2026', '...'),
    ('\u2018', "'"),
    ('\u2019', "'"),
]

for fpath in files:
    if not os.path.exists(fpath):
        print(f'SKIP: {fpath}')
        continue
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    new_content = content
    for bad, good in REPLACEMENTS:
        new_content = new_content.replace(bad, good)
    # Fallback: encode/decode round-trip to drop any remaining non-cp1252
    safe = new_content.encode('cp1252', errors='replace').decode('cp1252')
    with open(fpath, 'w', encoding='utf-8') as f:
        f.write(safe)
    status = 'FIXED' if safe != content else 'OK'
    print(f'{status}: {fpath}')

print('Done.')
