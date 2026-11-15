import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import bench


class BenchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.engine = bench.Engine()

    @classmethod
    def tearDownClass(cls):
        cls.engine.close()

    def test_quantum_state_and_complete_moves(self):
        moves = ['e3/4', 'e5', 'Nf3', 'Nc6', 'Bb5/c4', 'Ne7/f6', '0-0/']
        state = self.engine.call(op='state', history=moves)
        self.assertEqual(state['state_count'], 16)
        self.assertEqual(state['to_move'], 'black')
        self.assertTrue(any('/' in m for m in state['legal_moves']))
        for move in state['legal_moves']:
            # Each advertised action must be accepted by the authoritative engine.
            self.engine.call(op='state', history=moves+[move])

    def test_rejections_forfeit_without_advancing(self):
        config = {'max_plies': 4, 'max_attempts': 3}
        a, b = {'name':'a'}, {'name':'b'}
        with tempfile.TemporaryDirectory() as tmp, patch('bench.completion', return_value=({'content':'illegal'}, {})):
            path = Path(tmp)/'calls.jsonl'
            result = bench.play(self.engine, config, a, b, [], 1, path)
            self.assertEqual(result['termination'], 'invalid_move_forfeit')
            self.assertEqual(result['score'], 0)
            self.assertEqual(result['moves'], [])
            events = [json.loads(l) for l in path.read_text().splitlines()]
            self.assertEqual(len(events), 3)
            self.assertEqual(len({e['state_hash'] for e in events}), 1)

    def test_api_failure_is_not_a_loss(self):
        with tempfile.TemporaryDirectory() as tmp, patch('bench.completion', side_effect=TimeoutError):
            with self.assertRaises(RuntimeError):
                bench.play(self.engine, {}, {'name':'a'}, {'name':'b'}, [], 1, Path(tmp)/'calls.jsonl')

    def test_schedule_resume_and_manifest_guard(self):
        config = {'players':[{'name':'a','kind':'random'},{'name':'b','kind':'random'}], 'max_plies':2}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            bench.run(config, path, self.engine)
            bench.audit(path, self.engine)
            records = [p.read_bytes() for p in sorted(path.glob('game-*.json'))]
            bench.run(config, path, self.engine)
            self.assertEqual(records, [p.read_bytes() for p in sorted(path.glob('game-*.json'))])
            summary = json.loads((path/'standings.json').read_text())
            self.assertEqual(summary['completed'], 2)
            self.assertTrue(summary['placement_complete'])
            config['max_plies'] = 3
            with self.assertRaises(ValueError):
                bench.run(config, path, self.engine)

    def test_checkmate_is_not_cutoff_draw(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = bench.play(self.engine, {'max_plies':1}, {'name':'a'}, {'name':'b'},
                                ['f3','e5','g4','Qh4#'], 1, Path(tmp)/'calls.jsonl')
            self.assertEqual(result['score'], 0)
            self.assertEqual(result['termination'], 'black_win')

    def test_http_adapter_end_to_end(self):
        captured = []
        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                captured.append(request)
                state = next(json.loads(m['content']) for m in reversed(request['messages']) if m['role'] == 'user')
                response = {'choices':[{'message':{'role':'assistant','content':None,'tool_calls':[{'id':'call_1','type':'function','function':{'name':'submit_move','arguments':json.dumps({'move':state['legal_moves'][0]})}}]}}],
                            'usage':{'total_tokens':12,'cost':.001}}
                data = json.dumps(response).encode()
                self.send_response(200)
                self.send_header('Content-Type','application/json')
                self.end_headers()
                self.wfile.write(data)
            def log_message(self, *args):
                pass
        server = HTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            player = {'name':'mock','model':'mock-model','api_key_env':'MOCK_KEY',
                      'base_url':f'http://127.0.0.1:{server.server_port}/v1',
                      'parameters':{'temperature':0}}
            with tempfile.TemporaryDirectory() as tmp, patch.dict('os.environ', {'MOCK_KEY':'dummy'}):
                result = bench.play(self.engine, {'max_plies':2}, player, player, [], 1, Path(tmp)/'calls.jsonl')
                self.assertEqual(len(result['moves']),2)
                self.assertEqual(result['termination'],'move_limit')
                self.assertEqual(captured[0]['model'],'mock-model')
                self.assertEqual(captured[0]['temperature'],0)
                self.assertEqual(captured[0]['tool_choice']['function']['name'], 'submit_move')
                self.assertIn('legal_moves',captured[0]['messages'][-1]['content'])
        finally:
            server.shutdown()
            server.server_close()
            thread.join()

    def test_batch_ratings_symmetric(self):
        r = self.engine.call(op='ratings', names=['a','b'], results=[{'white':'a','black':'b','score':1}])
        self.assertGreater(r[0]['rating'],1500)
        self.assertAlmostEqual(r[0]['rating']+r[1]['rating'],3000)


if __name__ == '__main__':
    unittest.main()
