import http from 'k6/http';
import { check, sleep } from 'k6';

export const options = {
  stages: [
    { duration: '30s', target: 25 },
    { duration: '45s', target: 50 },
    { duration: '60s', target: 100 },
    { duration: '30s', target: 0 },
  ],
};

export default function () {
  const res = http.get(`${__ENV.BASE_URL}/ping`);
  check(res, { 'ping status 200': (r) => r.status === 200 });
  sleep(1);
}