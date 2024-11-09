// script.js

let temperature = 0;

async function startReactor() {
  const response = await fetch('http://localhost:8080/start', {
    method: 'POST',
  });
  const data = await response.json();
  console.log(data);
}

async function updateReactor(temperature) {
  const response = await fetch('http://localhost:8080/update', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ temperature }),
  });
  const data = await response.json();
  console.log(data);
}

async function getReactorState() {
  const response = await fetch('http://localhost:8080/state');
  const data = await response.json();
  console.log(data);
  return data;
}

function updateTemperature(value) {
  temperature = parseFloat(value);
  document.getElementById('temperatureValue').innerText = value;
  updateReactor(temperature);
}

// Periodically update reactor state display
setInterval(async () => {
  const state = await getReactorState();
  document.getElementById('reactorState').innerText = JSON.stringify(state, null, 2);
}, 1000);
