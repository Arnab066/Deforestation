const dropZoneS2 = document.getElementById('drop-zone-s2');
const fileInputS2 = document.getElementById('file-s2');
const previewS2 = document.getElementById('preview-s2');

const dropZoneS1 = document.getElementById('drop-zone-s1');
const fileInputS1 = document.getElementById('file-s1');
const previewS1 = document.getElementById('preview-s1');

const analyzeBtn = document.getElementById('analyze-btn');
const loader = document.getElementById('loader');
const results = document.getElementById('results');
const riskLevel = document.getElementById('risk-level');
const outputImg = document.getElementById('output-img');

let selectedFileS2 = null;
let selectedFileS1 = null;

// S2 Upload logic
dropZoneS2.addEventListener('click', () => fileInputS2.click());
dropZoneS2.addEventListener('dragover', (e) => { e.preventDefault(); dropZoneS2.classList.add('dragover'); });
dropZoneS2.addEventListener('dragleave', () => dropZoneS2.classList.remove('dragover'));
dropZoneS2.addEventListener('drop', (e) => {
    e.preventDefault(); dropZoneS2.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], 's2');
});
fileInputS2.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0], 's2');
});

// S1 Upload logic
dropZoneS1.addEventListener('click', () => fileInputS1.click());
dropZoneS1.addEventListener('dragover', (e) => { e.preventDefault(); dropZoneS1.classList.add('dragover'); });
dropZoneS1.addEventListener('dragleave', () => dropZoneS1.classList.remove('dragover'));
dropZoneS1.addEventListener('drop', (e) => {
    e.preventDefault(); dropZoneS1.classList.remove('dragover');
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0], 's1');
});
fileInputS1.addEventListener('change', (e) => {
    if (e.target.files.length) handleFile(e.target.files[0], 's1');
});

function handleFile(file, type) {
    if (!file.type.startsWith('image/')) {
        alert('Please select an image file.');
        return;
    }
    
    const reader = new FileReader();
    reader.onload = (e) => {
        if (type === 's2') {
            selectedFileS2 = file;
            previewS2.src = e.target.result;
            previewS2.style.display = 'block';
            dropZoneS2.querySelector('p').style.display = 'none';
            analyzeBtn.disabled = false; // Enable if S2 is present
        } else {
            selectedFileS1 = file;
            previewS1.src = e.target.result;
            previewS1.style.display = 'block';
            dropZoneS1.querySelector('p').style.display = 'none';
        }
    };
    reader.readAsDataURL(file);
}

analyzeBtn.addEventListener('click', async () => {
    if (!selectedFileS2) return;
    
    const formData = new FormData();
    formData.append('image_s2', selectedFileS2);
    if (selectedFileS1) {
        formData.append('image_s1', selectedFileS1);
    }
    
    analyzeBtn.disabled = true;
    loader.style.display = 'block';
    results.style.display = 'none';
    
    try {
        const response = await fetch('/predict', {
            method: 'POST',
            body: formData
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.error || 'Server error');
        }
        
        riskLevel.textContent = data.risk;
        riskLevel.style.color = data.risk === 'High' ? '#e74c3c' : (data.risk === 'Medium' ? '#f1c40f' : '#2ecc71');
        outputImg.src = data.output_image.startsWith('data:') ? data.output_image : ('/static/' + data.output_image + '?t=' + new Date().getTime());
        
        results.style.display = 'block';
    } catch (error) {
        alert('Error: ' + error.message);
    } finally {
        analyzeBtn.disabled = false;
        loader.style.display = 'none';
    }
});
