document.addEventListener('DOMContentLoaded', () => {
    // Custom Cursor Glow Follow Effect
    const cursorGlow = document.getElementById('cursor-glow');
    if (cursorGlow) {
        document.addEventListener('mousemove', (e) => {
            cursorGlow.style.left = e.clientX + 'px';
            cursorGlow.style.top = e.clientY + 'px';
        });
        
        document.addEventListener('mousedown', () => {
            cursorGlow.style.width = '200px';
            cursorGlow.style.height = '200px';
        });
        
        document.addEventListener('mouseup', () => {
            cursorGlow.style.width = '300px';
            cursorGlow.style.height = '300px';
        });
    }

    // Scroll Reveal
    const observer = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                entry.target.classList.add('active');
            }
        });
    }, { threshold: 0.1, rootMargin: "0px 0px -50px 0px" });
    document.querySelectorAll('.reveal').forEach(el => observer.observe(el));

    // Interactive Terminal
    const termInput = document.getElementById('term-input');
    const termOutput = document.getElementById('term-output');

    if (termInput && termOutput) {
        termInput.addEventListener('keypress', function(e) {
            if (e.key === 'Enter') {
                const cmd = termInput.value.trim();
                if (cmd) {
                    processCommand(cmd);
                    termInput.value = '';
                }
            }
        });
    }

    function processCommand(cmd) {
        // Echo command
        const cmdLine = document.createElement('div');
        cmdLine.className = 'term-line';
        cmdLine.innerHTML = `<span class="term-prompt">engem@ai:~$</span> <span class="term-text">${cmd}</span>`;
        termOutput.appendChild(cmdLine);

        // Response
        const resLine = document.createElement('div');
        resLine.className = 'term-output';
        
        let response = '';
        const lowerCmd = cmd.toLowerCase();

        if (lowerCmd === 'help') {
            response = 'Available commands: help, about, features, clear, engem';
        } else if (lowerCmd === 'about') {
            response = 'EnGem: Multi-modal agentic platform powered by Gemini.';
        } else if (lowerCmd === 'features') {
            response = 'Skills: use_browser, run_python, generate_image, generate_video, run_notebook...';
        } else if (lowerCmd === 'engem') {
            response = 'Intelligence Accelerated. Welcome to the future.';
        } else if (lowerCmd === 'clear') {
            termOutput.innerHTML = '';
            return;
        } else if (lowerCmd.startsWith('echo ')) {
            response = cmd.substring(5);
        } else {
            response = `Command not found: ${cmd}. Type 'help' for available commands.`;
        }

        resLine.innerText = response;
        termOutput.appendChild(resLine);
        
        // Auto scroll
        const body = document.querySelector('.terminal-body');
        body.scrollTop = body.scrollHeight;
    }

    // Canvas Background Animation
    const canvas = document.getElementById('canvas-bg');
    if (canvas) {
        const ctx = canvas.getContext('2d');
        let width, height;
        let particles = [];

        function init() {
            width = canvas.width = window.innerWidth;
            height = canvas.height = window.innerHeight;
            particles = [];
            for (let i = 0; i < 50; i++) {
                particles.push({
                    x: Math.random() * width,
                    y: Math.random() * height,
                    vx: (Math.random() - 0.5) * 0.5,
                    vy: (Math.random() - 0.5) * 0.5,
                    size: Math.random() * 2 + 0.5,
                    alpha: Math.random() * 0.5 + 0.1
                });
            }
        }

        function animate() {
            ctx.clearRect(0, 0, width, height);
            
            particles.forEach(p => {
                p.x += p.vx;
                p.y += p.vy;
                
                if (p.x < 0 || p.x > width) p.vx *= -1;
                if (p.y < 0 || p.y > height) p.vy *= -1;
                
                ctx.beginPath();
                ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);
                ctx.fillStyle = `rgba(0, 255, 204, ${p.alpha})`;
                ctx.fill();
            });

            // Draw connections
            for (let i = 0; i < particles.length; i++) {
                for (let j = i + 1; j < particles.length; j++) {
                    const dx = particles[i].x - particles[j].x;
                    const dy = particles[i].y - particles[j].y;
                    const dist = Math.sqrt(dx * dx + dy * dy);
                    
                    if (dist < 150) {
                        ctx.beginPath();
                        ctx.moveTo(particles[i].x, particles[i].y);
                        ctx.lineTo(particles[j].x, particles[j].y);
                        ctx.strokeStyle = `rgba(191, 0, 255, ${0.1 * (1 - dist/150)})`;
                        ctx.stroke();
                    }
                }
            }
            
            requestAnimationFrame(animate);
        }

        init();
        animate();
        window.addEventListener('resize', init);
    }
});
