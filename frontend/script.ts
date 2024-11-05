// script.js
let temperature = 0;

function updateTemperature(value: string) {
  temperature = parseFloat(value);
  document.getElementById('temperatureValue')!.innerText = value;
  updateReactor(temperature);
}


// Periodically update reactor state display
// PLAN: Debounce reactor calls; could even optimize the debounce
setInterval(async () => {
  const state = await getReactorState();
  document.getElementById('reactorState')!.innerText = JSON.stringify(state, null, 2);
}, 1000);
