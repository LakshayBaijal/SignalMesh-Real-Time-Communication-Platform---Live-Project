import http from 'k6/http';
import { check, sleep } from 'k6';
import { Rate } from 'k6/metrics';

const BASE_URL = __ENV.BASE_URL || 'https://signal-mesh.onrender.com';
const BENCH_USERNAME = __ENV.BENCH_USERNAME || 'bench_login_user';
const BENCH_PASSWORD = __ENV.BENCH_PASSWORD || 'BenchPass123!';

const requestChecks = new Rate('http_checks');

export const options = {
  scenarios: {
    ping_latency: {
      executor: 'constant-vus',
      vus: 1,
      duration: '20s',
      exec: 'pingScenario',
    },
    login_throughput: {
      executor: 'constant-vus',
      vus: 1,
      duration: '20s',
      startTime: '20s',
      exec: 'loginScenario',
    },
  },
};

function joinUrl(base, path) {
  const left = base.endsWith('/') ? base.slice(0, -1) : base;
  const right = path.startsWith('/') ? path : `/${path}`;
  return `${left}${right}`;
}

export function setup() {
  const url = joinUrl(BASE_URL, '/signup');
  const payload = JSON.stringify({ username: BENCH_USERNAME, password: BENCH_PASSWORD });
  const params = { headers: { 'Content-Type': 'application/json' } };

  const response = http.post(url, payload, params);
  check(response, {
    'setup signup accepted (200 or 400)': (r) => r.status === 200 || r.status === 400,
  });

  return { ready: true };
}

export function pingScenario() {
  const response = http.get(joinUrl(BASE_URL, '/ping'));
  const ok = check(response, {
    'ping status 200': (r) => r.status === 200,
  });
  requestChecks.add(ok);
  sleep(0.2);
}

export function loginScenario() {
  const payload = JSON.stringify({ username: BENCH_USERNAME, password: BENCH_PASSWORD });
  const params = { headers: { 'Content-Type': 'application/json' } };

  const response = http.post(joinUrl(BASE_URL, '/login'), payload, params);
  const ok = check(response, {
    'login status 200': (r) => r.status === 200,
    'login token exists': (r) => {
      try {
        return Boolean(r.json('token'));
      } catch (error) {
        return false;
      }
    },
  });
  requestChecks.add(ok);
  sleep(0.1);
}
