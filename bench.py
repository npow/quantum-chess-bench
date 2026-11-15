#!/usr/bin/env python3
"""Quantum Chess Bench: standard-library tournament runner."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
PROTOCOL = 'quantum-chess-bench-v1'
SYSTEM = ('You play quantum chess. Win using your own reasoning. The engine is the authority. '
          'Select exactly one string from legal_moves, including any collapse suffix. '
          'Return only JSON {"move":"the exact move"}. Do not use external engines.\n\n' +
          (ROOT / 'RULES.md').read_text())


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


class Engine:
    def __init__(self):
        subprocess.run(['cargo', 'build', '--release', '--locked', '-q'], cwd=ROOT, check=True)
        self.p = subprocess.Popen([str(ROOT / 'target/release/qbench-engine')], stdin=subprocess.PIPE,
                                  stdout=subprocess.PIPE, text=True)

    def call(self, **data):
        self.p.stdin.write(json.dumps(data) + '\n')
        self.p.stdin.flush()
        line = self.p.stdout.readline()
        if not line:
            raise RuntimeError('rules engine exited')
        result = json.loads(line)
        if 'error' in result:
            raise ValueError(result['error'])
        return result['ok']

    def close(self):
        self.p.stdin.close()
        self.p.wait(timeout=10)


def parse_move(message):
    content = message.get('content') or ''
    if not isinstance(content, str):
        raise ValueError('expected text content')
    calls = message.get('tool_calls') or []
    if calls:
        if len(calls) != 1 or calls[0]['function']['name'] != 'submit_move':
            raise ValueError('expected one submit_move call')
        return json.loads(calls[0]['function']['arguments'])['move']
    try:
        return json.loads(content)['move']
    except (ValueError, KeyError, TypeError):
        return content.strip()


def completion(player, messages, timeout):
    # Credentials never enter saved configuration or request records.
    key = os.environ[player.get('api_key_env', 'OPENROUTER_API_KEY')]
    body = dict(player.get('parameters', {}))
    body.update(model=player['model'], messages=messages, stream=False)
    req = urllib.request.Request(player.get('base_url', 'https://openrouter.ai/api/v1').rstrip('/') +
                                 '/chat/completions', data=json.dumps(body).encode(),
                                 headers={'Authorization': 'Bearer ' + key, 'Content-Type': 'application/json'})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        result = json.load(response)
    return result['choices'][0]['message'], result.get('usage', {})


def write_json(path, data):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data, indent=2) + '\n')
    tmp.replace(path)


def play(engine, config, white, black, opening, seed, log_path):
    history = list(opening)
    chats = [[], []]
    rng = random.Random(seed)
    seen = {}
    with log_path.open('a') as log:
        while True:
            state = engine.call(op='state', history=history)
            status = state['status']
            if status != 'ongoing':
                score = {'white_win': 1, 'black_win': 0, 'stalemate': .5}[status]
                break
            key = state['position_key']
            seen[key] = seen.get(key, 0) + 1
            if seen[key] >= 3:
                score, status = .5, 'threefold_repetition'
                break
            if len(history) - len(opening) >= config.get('max_plies', 150):
                score, status = .5, 'move_limit'
                break
            side = 0 if state['to_move'] == 'white' else 1
            player = [white, black][side]
            # Full current state is always retained; prune only complete older turns.
            chat = chats[side]
            while len(json.dumps(chat)) > config.get('context_chars', 60000) and len(chat) >= 2:
                del chat[:2]
            chat.append({'role': 'user', 'content': json.dumps(state)})
            for attempt in range(config.get('max_attempts', 3)):
                started = time.monotonic()
                event = {'ply': len(history), 'player': player['name'], 'attempt': attempt + 1,
                         'state_hash': digest(state), 'legal_count': len(state['legal_moves'])}
                kind = player.get('kind', 'llm')
                try:
                    if kind == 'random':
                        message, usage = {'role': 'assistant', 'content': rng.choice(state['legal_moves'])}, {}
                    elif kind == 'engine':
                        move = engine.call(op='choose', history=history)['move']
                        message, usage = {'role': 'assistant', 'content': move}, {}
                    else:
                        message, usage = completion(player, [{'role': 'system', 'content': SYSTEM}] + chat,
                                                    config.get('timeout_seconds', 120))
                except Exception as exc:
                    # Infrastructure failures are incomplete games, never losses.
                    event.update(error=type(exc).__name__, elapsed_seconds=time.monotonic() - started)
                    log.write(json.dumps(event) + '\n'); log.flush()
                    raise RuntimeError(f"request failed for {player['name']}: {type(exc).__name__}") from exc
                chat.append({'role': 'assistant', 'content': message.get('content') or json.dumps(message.get('tool_calls'))})
                try:
                    move = parse_move(message)
                except (ValueError, KeyError, TypeError):
                    move = None
                accepted = isinstance(move, str) and move in state['legal_moves']
                event.update(response=message, usage=usage, move=move, accepted=accepted,
                             elapsed_seconds=time.monotonic() - started)
                # Input snapshots make transcripts independently auditable.
                event['state'] = state
                log.write(json.dumps(event) + '\n'); log.flush()
                if accepted:
                    history.append(move)
                    break
                chat.append({'role': 'user', 'content': f'Rejected move. Attempt {attempt+1}. Board unchanged; opponent has not moved. Select from legal_moves.'})
            else:
                score, status = (0 if side == 0 else 1), 'invalid_move_forfeit'
                break
    return {'white': white['name'], 'black': black['name'], 'score': score, 'termination': status,
            'opening': opening, 'moves': history, 'seed': seed, 'final_state_hash': digest(state)}


def validate(config):
    players = config['players']
    if len(players) < 2 or len({p['name'] for p in players}) != len(players):
        raise ValueError('provide at least two uniquely named players')
    for key in ('pairs', 'max_plies', 'max_attempts', 'timeout_seconds', 'context_chars'):
        if key in config and (type(config[key]) is not int or config[key] <= 0):
            raise ValueError(f'{key} must be a positive integer')
    for p in players:
        if p.get('kind', 'llm') not in ('llm', 'random', 'engine'):
            raise ValueError('kind must be llm, random, or engine')
        if p.get('kind', 'llm') == 'llm':
            if not p.get('model'):
                raise ValueError('LLM player requires model')
            os.environ[p.get('api_key_env', 'OPENROUTER_API_KEY')]
    return players


def run(config, output, engine):
    players = validate(config)
    output.mkdir(parents=True, exist_ok=True)
    identity = {'protocol': PROTOCOL, 'config': config, 'system_prompt': SYSTEM,
                'engine_source_hash': digest({str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
                    for folder in ('src', 'vendor/qchess/src', 'vendor/glicko2/src')
                    for p in sorted((ROOT / folder).glob('*.rs'))}),
                'runner_hash': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'build_hash': digest({p: (ROOT / p).read_text() for p in ('Cargo.toml', 'Cargo.lock', 'UPSTREAM.md')})}
    identity['hash'] = digest(identity)
    manifest = output / 'manifest.json'
    if manifest.exists() and json.loads(manifest.read_text()) != identity:
        raise ValueError('run configuration or code changed; use a new output directory')
    write_json(manifest, identity)
    schedule = []
    openings = config.get('openings', [[]])
    if not openings:
        raise ValueError('openings must not be empty')
    for i, a in enumerate(players):
        for b in players[i+1:]:
            for pair in range(config.get('pairs', 1)):
                opening = openings[pair % len(openings)]
                engine.call(op='state', history=opening)  # Validate before spending API credits.
                for w, bl in ((a, b), (b, a)):
                    schedule.append((w, bl, opening))
    results = []
    for index, (white, black, opening) in enumerate(schedule):
        path = output / f'game-{index:05d}.json'
        if path.exists():
            result = json.loads(path.read_text())
            if result.get('record_hash') != digest({k:v for k,v in result.items() if k != 'record_hash'}):
                raise ValueError(f'corrupted result: {path}')
        else:
            print(f'Game {index+1}/{len(schedule)}: {white["name"]} vs {black["name"]}', flush=True)
            result = play(engine, config, white, black, opening, config.get('seed', 1) + index,
                          output / f'game-{index:05d}.jsonl')
            result.update(game=index, manifest_hash=identity['hash'])
            result['record_hash'] = digest(result)
            write_json(path, result)
        results.append(result)
        ratings = engine.call(op='ratings', names=[p['name'] for p in players], results=results)
        summary = {'protocol': PROTOCOL, 'completed': len(results), 'scheduled': len(schedule),
                   'placement_complete': len(results) == len(schedule),
                   'results_hash': digest(results), 'ratings': sorted(ratings, key=lambda r:-r['rating']),
                   'metrics': metrics(output, results, players)}
        write_json(output / 'standings.json', summary)
    print(json.dumps(summary, indent=2))


def metrics(output, results, players):
    out = {p['name']: {'wins': 0, 'draws': 0, 'losses': 0, 'answered_attempts': 0,
           'rejected_attempts': 0, 'request_failures': 0, 'request_seconds': 0.0,
           'reported_tokens': 0, 'reported_cost_usd': 0.0, 'priced_attempts': 0} for p in players}
    for r in results:
        for name, score in ((r['white'], r['score']), (r['black'], 1-r['score'])):
            out[name][{1:'wins', .5:'draws', 0:'losses'}[score]] += 1
    for path in output.glob('game-*.jsonl'):
        for line in path.read_text().splitlines():
            e = json.loads(line)
            row = out[e['player']]
            row['request_seconds'] += e['elapsed_seconds']
            if 'error' in e:
                row['request_failures'] += 1
                continue
            row['answered_attempts'] += 1
            row['rejected_attempts'] += int(not e['accepted'])
            usage = e['usage']
            row['reported_tokens'] += usage.get('total_tokens', 0) or 0
            if usage.get('cost') is not None:
                row['reported_cost_usd'] += usage['cost']
                row['priced_attempts'] += 1
    return out


def audit(output, engine):
    manifest = json.loads((output/'manifest.json').read_text())
    if manifest['hash'] != digest({k:v for k,v in manifest.items() if k != 'hash'}):
        raise ValueError('manifest hash mismatch')
    results = []
    for path in sorted(output.glob('game-*.json')):
        r = json.loads(path.read_text())
        if r['record_hash'] != digest({k:v for k,v in r.items() if k != 'record_hash'}):
            raise ValueError(f'record hash mismatch: {path}')
        if r['manifest_hash'] != manifest['hash']:
            raise ValueError('result belongs to a different manifest')
        state = engine.call(op='state', history=r['moves'])
        if digest(state) != r['final_state_hash']:
            raise ValueError(f'replay mismatch: {path}')
        for line in path.with_suffix('.jsonl').read_text().splitlines():
            e = json.loads(line)
            if 'state' in e and digest(e['state']) != e['state_hash']:
                raise ValueError('turn snapshot hash mismatch')
            if e.get('accepted') and e['move'] not in e['state']['legal_moves']:
                raise ValueError('accepted action outside legal list')
        results.append(r)
    standings = json.loads((output/'standings.json').read_text())
    if standings['results_hash'] != digest(results) or standings['completed'] != len(results):
        raise ValueError('standings sample count/hash mismatch')
    print(f"Verified {len(results)}/{standings['scheduled']} game records and final-position replays.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'smoke.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'runs/smoke')
    parser.add_argument('--audit', action='store_true', help='verify saved records and replay final positions')
    parser.add_argument('--position', action='store_true', help='print example opening state and legal actions')
    args = parser.parse_args()
    engine = Engine()
    try:
        if args.audit:
            audit(args.output, engine)
        elif args.position:
            print(json.dumps(engine.call(op='state', history=['e3/4', 'e5', 'Nf3', 'Nc6', 'Bb5/c4', 'Ne7/f6', '0-0/']), indent=2))
        else:
            run(json.loads(args.config.read_text()), args.output, engine)
    finally:
        engine.close()


if __name__ == '__main__':
    main()
