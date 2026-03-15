document.addEventListener('DOMContentLoaded', function() {
    // Get DOM elements
    const messageInput = document.getElementById('messageInput');
    const detectBtn = document.getElementById('detectBtn');
    const randomBtn = document.getElementById('randomBtn');
    const clearBtn = document.getElementById('clearBtn');
    const pasteBtn = document.getElementById('pasteBtn');
    const loadingSpinner = document.getElementById('loadingSpinner');
    const resultContainer = document.getElementById('resultContainer');
    const charCount = document.getElementById('charCount');
    const charProgress = document.getElementById('charProgress');
    
    // Result elements
    const resultIcon = document.getElementById('resultIcon');
    const resultTitle = document.getElementById('resultTitle');
    const resultSubtitle = document.getElementById('resultSubtitle');
    const analyzedText = document.getElementById('analyzedText');
    const classification = document.getElementById('classification');
    const confidence = document.getElementById('confidence');
    const confidenceFill = document.getElementById('confidenceFill');
    const confidenceText = document.getElementById('confidenceText');
    const confidencePercent = document.getElementById('confidencePercent');
    const progressCircle = document.getElementById('progressCircle');
    const warningMessage = document.getElementById('warningMessage');
    const shareBtn = document.getElementById('shareBtn');
    const copyBtn = document.getElementById('copyBtn');
    
    // Navigation elements
    const mobileMenuToggle = document.getElementById('mobileMenuToggle');
    const navList = document.querySelector('.nav-list');
    const navLinks = document.querySelectorAll('.nav-link');
    const themeToggle = document.getElementById('themeToggle');
    
    // History elements
    const refreshHistoryBtn = document.getElementById('refreshHistory');
    const clearHistoryBtn = document.getElementById('clearHistory');
    const historyLoading = document.getElementById('historyLoading');
    const historyTable = document.getElementById('historyTable');
    const historyBody = document.getElementById('historyBody');
    const emptyHistory = document.getElementById('emptyHistory');
    const totalHistory = document.getElementById('totalHistory');
    const safeHistory = document.getElementById('safeHistory');
    const toxicHistory = document.getElementById('toxicHistory');
    
    // Sample messages for testing
    const sampleMessages = [
        { text: "You are such a hard worker!", type: "safe" },
        { text: "Great job on the project!", type: "safe" },
        { text: "You're an amazing person", type: "safe" },
        { text: "you are too black", type: "toxic" },
        { text: "kill yourself", type: "toxic" },
        { text: "you are fat and ugly", type: "toxic" },
        { text: "slut", type: "toxic" },
        { text: "prostitute", type: "toxic" },
        { text: "go back to africa", type: "toxic" },
        { text: "you're a bitch", type: "toxic" }
    ];
    
    // Initialize
    if (messageInput && detectBtn) {
        initEventListeners();
        setupAnimations();
        setupNavigation();
        setupHistory();
    }
    
    function setupNavigation() {
        // Mobile menu toggle
        if (mobileMenuToggle && navList) {
            mobileMenuToggle.addEventListener('click', () => {
                navList.classList.toggle('active');
                const icon = mobileMenuToggle.querySelector('i');
                if (icon) {
                    icon.classList.toggle('fa-bars');
                    icon.classList.toggle('fa-times');
                }
            });
        }
        
        // Navigation link active states
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                // Remove active class from all links
                navLinks.forEach(l => l.classList.remove('active'));
                // Add active class to clicked link
                e.target.closest('.nav-link').classList.add('active');
                
                // Close mobile menu
                if (navList && window.innerWidth <= 768) {
                    navList.classList.remove('active');
                    if (mobileMenuToggle) {
                        const icon = mobileMenuToggle.querySelector('i');
                        if (icon) {
                            icon.classList.add('fa-bars');
                            icon.classList.remove('fa-times');
                        }
                    }
                }
            });
        });
        
        // Smooth scroll for navigation links
        navLinks.forEach(link => {
            link.addEventListener('click', (e) => {
                e.preventDefault();
                const targetId = link.getAttribute('href').substring(1);
                const targetElement = document.getElementById(targetId);
                
                if (targetElement) {
                    const offset = 80; // Account for fixed navbar
                    const targetPosition = targetElement.offsetTop - offset;
                    
                    window.scrollTo({
                        top: targetPosition,
                        behavior: 'smooth'
                    });
                }
            });
        });
        
        // Update active nav link on scroll
        updateActiveNavLink();
        window.addEventListener('scroll', updateActiveNavLink);
        
        // Theme toggle
        if (themeToggle) {
            themeToggle.addEventListener('click', () => {
                toggleTheme();
            });
        }
        
        // Initialize theme from localStorage
        initializeTheme();
    }
    
    function initializeTheme() {
        const savedTheme = localStorage.getItem('theme') || 'light';
        if (savedTheme === 'dark') {
            document.body.classList.add('dark-theme');
            const icon = themeToggle.querySelector('i');
            if (icon) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            }
        }
    }
    
    function toggleTheme() {
        const body = document.body;
        const icon = themeToggle.querySelector('i');
        
        // Add switching animation
        themeToggle.classList.add('switching');
        
        // Toggle theme
        if (body.classList.contains('dark-theme')) {
            body.classList.remove('dark-theme');
            localStorage.setItem('theme', 'light');
            if (icon) {
                icon.classList.remove('fa-sun');
                icon.classList.add('fa-moon');
            }
            showToast('Switched to light theme', 'info');
        } else {
            body.classList.add('dark-theme');
            localStorage.setItem('theme', 'dark');
            if (icon) {
                icon.classList.remove('fa-moon');
                icon.classList.add('fa-sun');
            }
            showToast('Switched to dark theme', 'info');
        }
        
        // Remove animation class
        setTimeout(() => {
            themeToggle.classList.remove('switching');
        }, 500);
        
        // Update analytics if on analytics page
        // Analytics section removed
    }
    
    function setupHistory() {
        // Load history on page load
        loadHistory();
        
        // Refresh button
        if (refreshHistoryBtn) {
            refreshHistoryBtn.addEventListener('click', loadHistory);
        }
        
        // Clear history button
        if (clearHistoryBtn) {
            clearHistoryBtn.addEventListener('click', clearHistory);
        }
    }
    
    async function loadHistory() {
        if (!historyLoading || !historyBody || !emptyHistory) return;
        
        try {
            // Show loading
            historyLoading.style.display = 'flex';
            historyTable.style.display = 'none';
            emptyHistory.style.display = 'none';
            
            // Fetch history from API
            const response = await fetch('/history?limit=50');
            const data = await response.json();
            
            if (data.success) {
                displayHistory(data.history);
                updateHistoryStats(data.history);
                console.log('History loaded successfully:', data.history.length, 'items');
            } else {
                throw new Error(data.error || 'Failed to load history');
            }
        } catch (error) {
            console.error('Error loading history:', error);
            showToast('Failed to load history', 'error');
        } finally {
            // Hide loading
            historyLoading.style.display = 'none';
        }
    }
    
    function displayHistory(history) {
        if (!historyBody) return;
        
        historyBody.innerHTML = '';
        
        if (history.length === 0) {
            historyTable.style.display = 'none';
            emptyHistory.style.display = 'block';
            return;
        }
        
        historyTable.style.display = 'block';
        emptyHistory.style.display = 'none';
        
        history.forEach(item => {
            const historyItem = createHistoryItem(item);
            historyBody.appendChild(historyItem);
        });
    }
    
    function createHistoryItem(item) {
        const div = document.createElement('div');
        div.className = 'history-item';
        
        const time = new Date(item.timestamp).toLocaleString();
        const isToxic = item.is_toxic;
        
        div.innerHTML = `
            <div class="history-time">${time}</div>
            <div class="history-message" title="${item.original_text}">${item.original_text}</div>
            <div class="history-result ${isToxic ? 'toxic' : 'safe'}">${item.label}</div>
            <div class="history-confidence">${item.confidence}%</div>
            <div class="history-actions">
                <button class="action-btn" onclick="copyToClipboard('${item.original_text}')" title="Copy message">
                    <i class="fas fa-copy"></i>
                </button>
                <button class="action-btn" onclick="viewDetails(${item.id})" title="View details">
                    <i class="fas fa-eye"></i>
                </button>
            </div>
        `;
        
        return div;
    }
    
    function updateHistoryStats(history) {
        if (!totalHistory || !safeHistory || !toxicHistory) return;
        
        const total = history.length;
        const safe = history.filter(item => !item.is_toxic).length;
        const toxic = history.filter(item => item.is_toxic).length;
        
        totalHistory.textContent = total;
        safeHistory.textContent = safe;
        toxicHistory.textContent = toxic;
    }
    
    async function clearHistory() {
        if (!confirm('Are you sure you want to clear all detection history? This action cannot be undone.')) {
            return;
        }
        
        try {
            const response = await fetch('/clear-history', {
                method: 'DELETE'
            });
            
            const data = await response.json();
            
            if (data.success) {
                showToast('History cleared successfully', 'success');
                loadHistory(); // Reload history
            } else {
                throw new Error(data.error || 'Failed to clear history');
            }
        } catch (error) {
            console.error('Error clearing history:', error);
            showToast('Failed to clear history', 'error');
        }
    }
    
    
    
    function updateActiveNavLink() {
        const sections = document.querySelectorAll('section[id]');
        const scrollPosition = window.scrollY + 100;
        
        sections.forEach(section => {
            const sectionTop = section.offsetTop;
            const sectionHeight = section.offsetHeight;
            const sectionId = section.getAttribute('id');
            
            if (scrollPosition >= sectionTop && scrollPosition < sectionTop + sectionHeight) {
                navLinks.forEach(link => {
                    link.classList.remove('active');
                    if (link.getAttribute('href') === `#${sectionId}`) {
                        link.classList.add('active');
                    }
                });
            }
        });
    }
    
    function initEventListeners() {
        // Character counter
        messageInput.addEventListener('input', updateCharCount);
        
        // Button click handlers
        detectBtn.addEventListener('click', analyzeMessage);
        if (randomBtn) randomBtn.addEventListener('click', loadRandomMessage);
        if (clearBtn) clearBtn.addEventListener('click', clearInput);
        if (pasteBtn) pasteBtn.addEventListener('click', pasteFromClipboard);
        if (shareBtn) shareBtn.addEventListener('click', shareResult);
        if (copyBtn) copyBtn.addEventListener('click', copyResult);
        
        // Keyboard shortcuts
        messageInput.addEventListener('keydown', handleKeyPress);
        
        // Auto-resize textarea
        messageInput.addEventListener('input', autoResizeTextarea);
        
        // Button ripple effects
        setupRippleEffects();
    }
    
    function updateCharCount() {
        const count = messageInput.value.length;
        const maxLength = 500;
        const percentage = (count / maxLength) * 100;
        
        if (charCount) charCount.textContent = count;
        if (charProgress) charProgress.style.width = `${Math.min(percentage, 100)}%`;
        
        // Color coding based on character count
        if (count > 450) {
            if (charCount) charCount.style.color = '#ef4444';
            if (charProgress) charProgress.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
        } else if (count > 350) {
            if (charCount) charCount.style.color = '#f59e0b';
            if (charProgress) charProgress.style.background = 'linear-gradient(90deg, #f59e0b, #d97706)';
        } else {
            if (charCount) charCount.style.color = '#4ade80';
            if (charProgress) charProgress.style.background = 'linear-gradient(90deg, #4ade80, #22c55e)';
        }
    }
    
    function handleKeyPress(e) {
        if (e.key === 'Enter' && e.ctrlKey) {
            e.preventDefault();
            analyzeMessage();
        } else if (e.key === 'Escape') {
            clearInput();
        }
    }
    
    function autoResizeTextarea() {
        messageInput.style.height = 'auto';
        messageInput.style.height = Math.min(messageInput.scrollHeight, 200) + 'px';
    }
    
    function setupRippleEffects() {
        const buttons = document.querySelectorAll('.detect-btn');
        buttons.forEach(button => {
            button.addEventListener('click', function(e) {
                const ripple = document.createElement('span');
                ripple.className = 'btn-ripple';
                
                const rect = this.getBoundingClientRect();
                const size = Math.max(rect.width, rect.height);
                const x = e.clientX - rect.left - size / 2;
                const y = e.clientY - rect.top - size / 2;
                
                ripple.style.width = ripple.style.height = size + 'px';
                ripple.style.left = x + 'px';
                ripple.style.top = y + 'px';
                
                this.appendChild(ripple);
                
                setTimeout(() => ripple.remove(), 600);
            });
        });
    }
    
    function setupAnimations() {
        // Add entrance animations to elements
        const observerOptions = {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        };
        
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.animation = 'fadeInUp 0.6s ease forwards';
                }
            });
        }, observerOptions);
        
        document.querySelectorAll('.feature-card, .category-card').forEach(card => {
            observer.observe(card);
        });
    }
    
    async function analyzeMessage() {
        const message = messageInput.value.trim();
        
        if (!message) {
            showToast('Please enter a message to analyze', 'warning');
            return;
        }
        
        // Show loading
        showLoading(true);
        
        try {
            const response = await fetch('/predict', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: message })
            });
            
            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }
            
            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }
            
            // Display result
            displayResults(result);
            
            // Refresh history after successful detection - with delay to ensure database is updated
            setTimeout(() => {
                if (refreshHistoryBtn) {
                    loadHistory();
                }
            }, 500); // Wait 500ms for database to be fully updated
            
            showToast('Analysis completed successfully!', 'success');
            
        } catch (error) {
            console.error('Analysis error:', error);
            showToast('Failed to analyze message. Please try again.', 'error');
        } finally {
            showLoading(false);
        }
    }
    
    function animateLoadingSteps() {
        const steps = document.querySelectorAll('.step');
        if (steps.length > 0) {
            steps.forEach((step, index) => {
                setTimeout(() => {
                    steps.forEach(s => s.classList.remove('active'));
                    step.classList.add('active');
                }, index * 500);
            });
        }
    }
    
    function showLoading(show) {
        if (show) {
            if (loadingSpinner) loadingSpinner.classList.remove('hidden');
            detectBtn.disabled = true;
            detectBtn.innerHTML = '<i class="fas fa-spinner fa-spin"></i><span>Analyzing...</span>';
        } else {
            if (loadingSpinner) loadingSpinner.classList.add('hidden');
            detectBtn.disabled = false;
            detectBtn.innerHTML = '<i class="fas fa-search"></i><span>Analyze Message</span>';
        }
    }
    
    function hideResults() {
        if (resultContainer) resultContainer.classList.add('hidden');
    }
    
    function displayResults(result) {
        // Update result header
        updateResultHeader(result);
        
        // Update analyzed text
        if (analyzedText) analyzedText.textContent = result.text;
        
        // Update classification
        updateClassification(result);
        
        // Update confidence
        updateConfidence(result);
        
        // Show/hide warning
        updateWarning(result);
        
        // Show results with animation
        if (resultContainer) resultContainer.classList.remove('hidden');
        
        // Scroll to results
        setTimeout(() => {
            if (resultContainer) {
                resultContainer.scrollIntoView({ 
                    behavior: 'smooth', 
                    block: 'nearest' 
                });
            }
        }, 100);
    }
    
    function updateResultHeader(result) {
        const isToxic = result.is_toxic || result.is_cyberbullying;
        
        if (resultTitle) {
            if (isToxic) {
                resultTitle.textContent = 'Toxic Content Detected';
                if (resultSubtitle) resultSubtitle.textContent = 'Warning: Harmful content identified';
                if (resultIcon) {
                    resultIcon.className = 'fas fa-exclamation-triangle';
                    resultIcon.style.color = '#ef4444';
                }
            } else {
                resultTitle.textContent = 'Safe Message';
                if (resultSubtitle) resultSubtitle.textContent = 'No harmful content detected';
                if (resultIcon) {
                    resultIcon.className = 'fas fa-check-circle';
                    resultIcon.style.color = '#4ade80';
                }
            }
        }
    }
    
    function updateClassification(result) {
        const isToxic = result.is_toxic || result.is_cyberbullying;
        
        if (classification) {
            const resultLabel = classification.querySelector('.result-label');
            if (resultLabel) {
                resultLabel.textContent = result.label;
                resultLabel.className = `result-label ${isToxic ? 'danger' : 'safe'}`;
            }
            
            // Update confidence bar
            const confPercentage = parseFloat(result.confidence);
            if (confidenceFill) {
                confidenceFill.style.width = `${confPercentage}%`;
                if (confidenceText) confidenceText.textContent = `${result.confidence}%`;
                
                if (isToxic) {
                    confidenceFill.style.background = 'linear-gradient(90deg, #ef4444, #dc2626)';
                    if (confidenceText) confidenceText.style.color = '#ef4444';
                } else {
                    confidenceFill.style.background = 'linear-gradient(90deg, #4ade80, #22c55e)';
                    if (confidenceText) confidenceText.style.color = '#4ade80';
                }
            }
        }
    }
    
    function updateConfidence(result) {
        const confPercentage = parseFloat(result.confidence);
        
        if (progressCircle && confidencePercent) {
            const circumference = 2 * Math.PI * 52;
            const offset = circumference - (confPercentage / 100) * circumference;
            
            progressCircle.style.strokeDashoffset = offset;
            confidencePercent.textContent = Math.round(confPercentage);
            
            // Update color based on confidence level
            if (confPercentage >= 80) {
                progressCircle.style.stroke = '#4ade80';
            } else if (confPercentage >= 60) {
                progressCircle.style.stroke = '#f59e0b';
            } else {
                progressCircle.style.stroke = '#ef4444';
            }
        }
        
        if (confidence) confidence.textContent = `${result.confidence}%`;
    }
    
    function updateWarning(result) {
        const isToxic = result.is_toxic || result.is_cyberbullying;
        
        if (warningMessage) {
            if (isToxic) {
                warningMessage.classList.remove('hidden');
            } else {
                warningMessage.classList.add('hidden');
            }
        }
    }
    
    function loadRandomMessage() {
        const randomIndex = Math.floor(Math.random() * sampleMessages.length);
        const message = sampleMessages[randomIndex];
        
        messageInput.value = message.text;
        updateCharCount();
        autoResizeTextarea();
        
        // Add animation effect
        messageInput.style.animation = 'pulse 0.5s ease';
        setTimeout(() => {
            messageInput.style.animation = '';
        }, 500);
        
        showToast(`Loaded ${message.type === 'safe' ? 'safe' : 'toxic'} sample message`, 'info');
    }
    
    function clearInput() {
        messageInput.value = '';
        updateCharCount();
        hideResults();
        messageInput.focus();
        
        // Add animation effect
        messageInput.style.animation = 'fadeIn 0.3s ease';
        setTimeout(() => {
            messageInput.style.animation = '';
        }, 300);
        
        showToast('Input cleared', 'info');
    }
    
    async function pasteFromClipboard() {
        try {
            const text = await navigator.clipboard.readText();
            messageInput.value = text;
            updateCharCount();
            autoResizeTextarea();
            messageInput.focus();
            
            showToast('Text pasted from clipboard', 'success');
        } catch (error) {
            showToast('Failed to paste from clipboard', 'error');
        }
    }
    
    function shareResult() {
        const resultText = `Analysis Result: ${resultTitle ? resultTitle.textContent : 'Unknown'} - Confidence: ${confidence ? confidence.textContent : 'Unknown'}`;
        
        if (navigator.share) {
            navigator.share({
                title: 'Toxic Comment Detection Result',
                text: resultText
            }).catch(() => {
                copyToClipboard(resultText);
            });
        } else {
            copyToClipboard(resultText);
        }
    }
    
    function copyResult() {
        const resultText = `Message: ${analyzedText ? analyzedText.textContent : 'Unknown'}\nClassification: ${resultTitle ? resultTitle.textContent : 'Unknown'}\nConfidence: ${confidence ? confidence.textContent : 'Unknown'}`;
        copyToClipboard(resultText);
    }
    
    
    function showToast(message, type = 'info') {
        const container = document.getElementById('toastContainer');
        if (!container) return;
        
        const toast = document.createElement('div');
        toast.className = `toast ${type}`;
        
        const icon = getToastIcon(type);
        toast.innerHTML = `
            <i class="fas ${icon}"></i>
            <span>${message}</span>
        `;
        
        container.appendChild(toast);
        
        // Auto remove after 3 seconds
        setTimeout(() => {
            toast.style.animation = 'slideOutRight 0.3s ease forwards';
            setTimeout(() => {
                if (container.contains(toast)) {
                    container.removeChild(toast);
                }
            }, 300);
        }, 3000);
    }
    
    function getToastIcon(type) {
        const icons = {
            success: 'fa-check-circle',
            error: 'fa-exclamation-circle',
            info: 'fa-info-circle',
            warning: 'fa-exclamation-triangle'
        };
        return icons[type] || icons.info;
    }
    
    // Add some interactive effects - DISABLED FOR MOBILE
    /*
    document.addEventListener('mousemove', (e) => {
        const shapes = document.querySelectorAll('.shape');
        const x = e.clientX / window.innerWidth;
        const y = e.clientY / window.innerHeight;
        
        shapes.forEach((shape, index) => {
            const speed = (index + 1) * 0.5;
            const xOffset = (x - 0.5) * speed * 20;
            const yOffset = (y - 0.5) * speed * 20;
            
            shape.style.transform = `translate(${xOffset}px, ${yOffset}px)`;
        });
    });
    */
    
    console.log('🖱️ Mousemove effects DISABLED for mobile compatibility');
    
    // Add keyboard navigation
    document.addEventListener('keydown', (e) => {
        // Ctrl/Cmd + Enter to analyze
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
            e.preventDefault();
            analyzeMessage();
        }
        
        // Ctrl/Cmd + K to clear
        if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
            e.preventDefault();
            clearInput();
        }
        
        // Ctrl/Cmd + R for random message
        if ((e.ctrlKey || e.metaKey) && e.key === 'r') {
            e.preventDefault();
            if (randomBtn) loadRandomMessage();
        }
    });
    
    // Add touch support for mobile - DISABLED TO FIX POPUP ISSUE
    // Temporarily disable all touch gestures to prevent popup spam
    let touchStartY = 0;
    let touchEndY = 0;
    let touchStartTime = 0;
    let touchEndTime = 0;
    let isSwipeGesture = false;
    
    // DISABLED: Touch event handlers causing popup issues
    /*
    document.addEventListener('touchstart', (e) => {
        // Only track touches on the main content area, not on buttons or inputs
        if (e.target.closest('button') || e.target.closest('input') || e.target.closest('textarea')) {
            return; // Ignore touches on interactive elements
        }
        
        touchStartY = e.changedTouches[0].screenY;
        touchStartTime = Date.now();
        isSwipeGesture = false;
    });
    
    document.addEventListener('touchmove', (e) => {
        // Detect if this is actually a swipe gesture
        if (Math.abs(e.changedTouches[0].screenY - touchStartY) > 10) {
            isSwipeGesture = true;
        }
    });
    
    document.addEventListener('touchend', (e) => {
        // Only process if it was a swipe gesture and not on interactive elements
        if (!isSwipeGesture || e.target.closest('button') || e.target.closest('input') || e.target.closest('textarea')) {
            return;
        }
        
        touchEndY = e.changedTouches[0].screenY;
        touchEndTime = Date.now();
        
        // Check if it was a quick swipe (not a long press)
        const touchDuration = touchEndTime - touchStartTime;
        if (touchDuration > 500) {
            return; // Ignore long presses
        }
        
        handleSwipe();
    });
    
    function handleSwipe() {
        const swipeThreshold = 80; // Increased threshold to prevent accidental triggers
        const diff = touchStartY - touchEndY;
        
        if (Math.abs(diff) > swipeThreshold) {
            if (diff > 0) {
                // Swipe up - analyze message (only if there's text)
                if (messageInput && messageInput.value.trim().length > 0) {
                    analyzeMessage();
                }
            } else {
                // Swipe down - clear input
                clearInput();
            }
        }
    }
    */
    
    // Add mobile-specific debug info
    console.log('📱 Mobile touch gestures DISABLED to prevent popup spam');
    console.log('🎯 Use buttons instead of swipe gestures on mobile');
    
    // Performance optimization - debounce resize events
    let resizeTimeout;
    window.addEventListener('resize', () => {
        clearTimeout(resizeTimeout);
        resizeTimeout = setTimeout(() => {
            autoResizeTextarea();
        }, 250);
    });
    
    // Add focus effects
    if (messageInput) {
        messageInput.addEventListener('focus', () => {
            messageInput.parentElement.classList.add('focused');
        });
        
        messageInput.addEventListener('blur', () => {
            messageInput.parentElement.classList.remove('focused');
        });
    }
    
    // Initialize page
    console.log('🚀 Toxic Comment Detection System initialized');
    // Remove welcome toast to prevent popup on mobile
    // showToast('Welcome! Enter a message to analyze for toxic content.', 'info');
});
// Add these OUTSIDE the DOMContentLoaded block, at the top or bottom of script.js

window.copyToClipboard = function(text) {
    navigator.clipboard.writeText(text).then(() => {
        showToast('Copied to clipboard', 'success');
    }).catch(() => {
        showToast('Failed to copy to clipboard', 'error');
    });
};

window.viewDetails = function(id) {
    showToast(`Viewing details for item #${id}`, 'info');
};

