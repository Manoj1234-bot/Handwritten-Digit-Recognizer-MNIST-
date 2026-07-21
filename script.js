// ----- DOM References -----
const canvas = document.getElementById('drawCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const errorMsg = document.getElementById('errorMsg');
const loader = document.getElementById('loader');
const predictionBox = document.getElementById('predictionBox');
const predictedDigit = document.getElementById('predictedDigit');
const confidenceText = document.getElementById('confidenceText');
const probabilityList = document.getElementById('probabilityList');

// ----- Canvas Setup -----
ctx.fillStyle = '#ffffff';
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.lineWidth = 18;
ctx.lineCap = 'round';
ctx.lineJoin = 'round';
ctx.strokeStyle = '#000000';

let isDrawing = false;
let lastX = 0;
let lastY = 0;

// ----- Get correct coordinates for mouse or touch -----
function getPos(e) {
  const rect = canvas.getBoundingClientRect();
  const scaleX = canvas.width / rect.width;
  const scaleY = canvas.height / rect.height;

  if (e.touches && e.touches.length > 0) {
    return {
      x: (e.touches[0].clientX - rect.left) * scaleX,
      y: (e.touches[0].clientY - rect.top) * scaleY,
    };
  }
  return {
    x: (e.clientX - rect.left) * scaleX,
    y: (e.clientY - rect.top) * scaleY,
  };
}

// ----- Drawing Handlers -----
function startDraw(e) {
  isDrawing = true;
  const pos = getPos(e);
  lastX = pos.x;
  lastY = pos.y;
}

function draw(e) {
  if (!isDrawing) return;
  e.preventDefault();

  const pos = getPos(e);
  ctx.beginPath();
  ctx.moveTo(lastX, lastY);
  ctx.lineTo(pos.x, pos.y);
  ctx.stroke();

  lastX = pos.x;
  lastY = pos.y;
}

function stopDraw() {
  isDrawing = false;
}

// Mouse events
canvas.addEventListener('mousedown', startDraw);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDraw);
canvas.addEventListener('mouseleave', stopDraw);

// Touch events (mobile/tablet drawing support)
canvas.addEventListener('touchstart', startDraw);
canvas.addEventListener('touchmove', draw);
canvas.addEventListener('touchend', stopDraw);

// ----- Clear Canvas -----
function clearCanvas() {
  ctx.fillStyle = '#ffffff';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  errorMsg.textContent = '';
  predictionBox.style.display = 'none';
  probabilityList.innerHTML = '';
}

// ----- Predict -----
async function predictDigit() {
  errorMsg.textContent = '';
  loader.classList.add('show');
  predictionBox.style.display = 'none';

  const imageData = canvas.toDataURL('image/png');

  try {
    const response = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image: imageData }),
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || 'Prediction failed.');
    }

    displayPrediction(data);
  } catch (err) {
    errorMsg.textContent = err.message;
  } finally {
    loader.classList.remove('show');
  }
}

// ----- Display Results -----
function displayPrediction(data) {
  predictedDigit.textContent = data.predicted_digit;
  confidenceText.textContent = `${data.confidence}% confidence`;
  predictionBox.style.display = 'block';

  probabilityList.innerHTML = '';

  const sortedEntries = Object.entries(data.all_probabilities);

  sortedEntries.forEach(([digit, prob]) => {
    const percent = (prob * 100).toFixed(1);
    const isTop = parseInt(digit) === data.predicted_digit;

    const row = document.createElement('div');
    row.className = 'prob-row';
    row.innerHTML = `
      <span class="prob-digit">${digit}</span>
      <div class="prob-track">
        <div class="prob-fill ${isTop ? 'top-prediction' : ''}" style="width: ${percent}%;"></div>
      </div>
      <span class="prob-percent">${percent}%</span>
    `;
    probabilityList.appendChild(row);
  });
}

// ----- Event Listeners -----
clearBtn.addEventListener('click', clearCanvas);
predictBtn.addEventListener('click', predictDigit);