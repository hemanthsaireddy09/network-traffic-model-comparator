document.getElementById('uploadForm').addEventListener('submit', function(e) {
    e.preventDefault();
    
    const fileInput = document.getElementById('file');
    const file = fileInput.files[0];
    
    if (!file) {
        alert('Please select a file');
        return;
    }
    
    const formData = new FormData();
    formData.append('file', file);
    
    // Show loading state
    document.querySelector('button[type="submit"]').disabled = true;
    document.querySelector('button[type="submit"]').innerHTML = 'Analyzing...';
    
    fetch('/upload', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        if (data.error) {
            alert(data.error);
            return;
        }
        
        // Show results section
        document.getElementById('results').style.display = 'block';
        
        // Display dataset statistics
        const statsDiv = document.getElementById('datasetStats');
        statsDiv.innerHTML = `
            <div class="metric-card">
                <p>Total Records: ${data.initial_stats.total_records}</p>
                <p>Number of Features: ${data.initial_stats.features.length - 1}</p>
            </div>
        `;
        
        // Display class distribution
        const distDiv = document.getElementById('classDistribution');
        let distHtml = '<div class="metric-card"><ul class="list-group">';
        for (const [className, count] of Object.entries(data.initial_stats.class_distribution)) {
            distHtml += `<li class="list-group-item d-flex justify-content-between align-items-center">
                ${className}
                <span class="badge bg-primary rounded-pill">${count}</span>
            </li>`;
        }
        distHtml += '</ul></div>';
        distDiv.innerHTML = distHtml;
        
        // Display accuracy plot
        const plotImg = document.getElementById('accuracyPlot');
        plotImg.src = `data:image/png;base64,${data.plot}`;
        
        // Display detailed results
        const resultsDiv = document.getElementById('algorithmResults');
        let resultsHtml = '<div class="row">';
        
        for (const [algorithm, metrics] of Object.entries(data.results)) {
            resultsHtml += `
                <div class="col-md-6 mb-3">
                    <div class="card">
                        <div class="card-header bg-light">
                            <h6 class="mb-0">${algorithm}</h6>
                        </div>
                        <div class="card-body">
                            <p><strong>Accuracy:</strong> <span class="metric-value">${metrics.accuracy.toFixed(2)}%</span></p>
                            <p><strong>Precision:</strong> <span class="metric-value">${metrics.precision.toFixed(2)}%</span></p>
                            <p><strong>Recall:</strong> <span class="metric-value">${metrics.recall.toFixed(2)}%</span></p>
                            <p><strong>F1 Score:</strong> <span class="metric-value">${metrics.f1.toFixed(2)}%</span></p>
                        </div>
                    </div>
                </div>
            `;
        }
        
        resultsHtml += '</div>';
        resultsDiv.innerHTML = resultsHtml;
    })
    .catch(error => {
        console.error('Error:', error);
        alert('An error occurred while processing the file');
    })
    .finally(() => {
        // Reset button state
        document.querySelector('button[type="submit"]').disabled = false;
        document.querySelector('button[type="submit"]').innerHTML = 'Analyze Traffic';
    });
});