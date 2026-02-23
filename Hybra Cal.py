import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import math
import random
import statistics
import re
import json
import threading
import requests
import sympy
from sympy import symbols, diff, integrate, limit, series, lambdify

# Optional plotting support
try:
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
    import numpy as np
    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False

class HybridCalculator:
    def __init__(self, root):
        self.root = root
        self.root.title("Hybra Cal - Ultimate Math & AI Suite")
        self.root.geometry("900x900")
        
        # API Key for AI functionality (empty by default)
        self.api_key = "" 
        
        # State variables
        self.angle_unit = "radians"
        self.current_mode = "Scientific"
        self.notes_content = "--- Quick Math Reference ---\nArithmetic: Basic operations\nAlgebra: Variable solving\nCalculus: Rates of change\nGeometry: Shapes & Angles\n"
        
        # UI Setup
        self.setup_ui()
        
    def setup_ui(self):
        # Result Display
        self.display_var = tk.StringVar(value="0")
        self.display = tk.Entry(self.root, textvariable=self.display_var, font=("Consolas", 28), 
                               justify='right', bd=15, relief="flat", bg="#f4f4f4")
        self.display.pack(fill="x", padx=20, pady=10)
        
        # Navigation Bar
        self.nav_frame = ttk.Frame(self.root)
        self.nav_frame.pack(fill="x", padx=10)
        
        modes = ["Scientific", "Algebra", "Statistics", "Graphing", "Notes", "AI Assistant"]
        for mode in modes:
            btn = ttk.Button(self.nav_frame, text=mode, command=lambda m=mode: self.change_mode(m))
            btn.pack(side="left", expand=True, fill="x")
            
        # Dynamic Content Frame
        self.main_container = ttk.Frame(self.root)
        self.main_container.pack(expand=True, fill="both", padx=10, pady=10)
        
        self.render_scientific_ui()

    def display_result(self, text):
        self.display_var.set(text)

    def add_to_input(self, val):
        current = self.display_var.get()
        if current == "0" or current == "Error":
            self.display_var.set(val)
        else:
            self.display_var.set(current + val)

    def clear_display(self):
        self.display_var.set("0")

    def change_mode(self, mode):
        # Save notes if leaving Notes mode
        if self.current_mode == "Notes" and hasattr(self, 'notes_area'):
            self.notes_content = self.notes_area.get("1.0", tk.END)
            
        self.current_mode = mode
        for widget in self.main_container.winfo_children():
            widget.destroy()
            
        if mode == "Scientific": self.render_scientific_ui()
        elif mode == "Algebra": self.render_algebra_ui()
        elif mode == "Statistics": self.render_stats_ui()
        elif mode == "Graphing": self.render_graphing_ui()
        elif mode == "Notes": self.render_notes_ui()
        elif mode == "AI Assistant": self.render_ai_ui()

    # --- Scientific Mode ---
    def render_scientific_ui(self):
        self.main_container.columnconfigure(tuple(range(10)), weight=1)
        buttons = [
            ('(', '(', 0, 0), (')', ')', 0, 1), ('C', self.clear_display, 0, 2), ('DEL', self.backspace, 0, 3), ('%', '/100', 0, 4), ('÷', '/', 0, 5),
            ('sin', 'math.sin(', 1, 0), ('cos', 'math.cos(', 1, 1), ('tan', 'math.tan(', 1, 2), ('asin', 'math.asin(', 1, 3), ('acos', 'math.acos(', 1, 4), ('atan', 'math.atan(', 1, 5),
            ('sinh', 'math.sinh(', 2, 0), ('cosh', 'math.cosh(', 2, 1), ('tanh', 'math.tanh(', 2, 2), ('log', 'math.log10(', 2, 3), ('ln', 'math.log(', 2, 4), ('exp', 'math.exp(', 2, 5),
            ('x²', '**2', 3, 0), ('x³', '**3', 3, 1), ('xʸ', '**', 3, 2), ('√', 'math.sqrt(', 3, 3), ('∛', '**(1/3)', 3, 4), ('!', 'math.factorial(', 3, 5),
            ('7', '7', 4, 0), ('8', '8', 4, 1), ('9', '9', 4, 2), ('×', '*', 4, 3), ('π', 'math.pi', 4, 4), ('e', 'math.e', 4, 5),
            ('4', '4', 5, 0), ('5', '5', 5, 1), ('6', '6', 5, 2), ('-', '-', 5, 3), ('Deg', 'DEG', 5, 4), ('Rad', 'RAD', 5, 5),
            ('1', '1', 6, 0), ('2', '2', 6, 1), ('3', '3', 6, 2), ('+', '+', 6, 3), ('Rand', 'RAND', 6, 4), ('abs', 'math.fabs(', 6, 5),
            ('0', '0', 7, 0), ('.', '.', 7, 1), ('=', 'CALC', 7, 2)
        ]
        
        for text, cmd, r, c in buttons:
            if cmd == 'CALC': action = self.calculate_scientific
            elif cmd == 'DEG': action = lambda: self.set_angle_units('degrees')
            elif cmd == 'RAD': action = lambda: self.set_angle_units('radians')
            elif cmd == 'RAND': action = lambda: self.display_var.set(str(random.random()))
            elif callable(cmd): action = cmd
            else: action = lambda v=cmd: self.add_to_input(v)
            
            btn = ttk.Button(self.main_container, text=text, command=action)
            btn.grid(row=r, column=c, sticky="nsew", padx=2, pady=2, ipady=10)

    def backspace(self):
        current = self.display_var.get()
        self.display_var.set(current[:-1] if len(current) > 1 else "0")

    def set_angle_units(self, unit):
        self.angle_unit = unit
        messagebox.showinfo("Unit Changed", f"Trigonometry set to {unit}")

    def calculate_scientific(self):
        expr = self.display_var.get()
        try:
            if self.angle_unit == "degrees":
                for func in ['sin', 'cos', 'tan', 'asin', 'acos', 'atan']:
                    expr = expr.replace(f"math.{func}(", f"math.{func}(math.radians(")
            
            open_c = expr.count('(')
            close_c = expr.count(')')
            if open_c > close_c: expr += ')' * (open_c - close_c)

            res = eval(expr, {"math": math, "__builtins__": None})
            self.display_var.set(str(res))
        except Exception:
            self.display_var.set("Error")

    # --- Algebra Mode ---
    def render_algebra_ui(self):
        ttk.Label(self.main_container, text="Symbolic Math Engine (Variable: x)", font=("Arial", 12, "bold")).pack(pady=10)
        self.alg_input = ttk.Entry(self.main_container, font=("Consolas", 16))
        self.alg_input.pack(fill="x", padx=20, pady=5)
        self.alg_input.insert(0, "x**2 + 2*x + 1")
        
        grid = ttk.Frame(self.main_container)
        grid.pack(pady=20)
        
        ops = [
            ("Simplify", self.alg_op("simplify")), ("Expand", self.alg_op("expand")),
            ("Factor", self.alg_op("factor")), ("Solve (f=0)", self.alg_op("solve")),
            ("Derive", self.alg_op("diff")), ("Integrate", self.alg_op("integrate")),
            ("Limit (x→0)", self.alg_op("limit")), ("Series", self.alg_op("series"))
        ]
        
        for i, (txt, cmd) in enumerate(ops):
            ttk.Button(grid, text=txt, command=cmd).grid(row=i//2, column=i%2, padx=10, pady=5, ipadx=20, ipady=5)

    def alg_op(self, op_type):
        def command():
            try:
                x = symbols('x')
                expr = sympy.sympify(self.alg_input.get().replace('^', '**'))
                if op_type == "simplify": res = sympy.simplify(expr)
                elif op_type == "expand": res = sympy.expand(expr)
                elif op_type == "factor": res = sympy.factor(expr)
                elif op_type == "solve": res = sympy.solve(expr, x)
                elif op_type == "diff": res = sympy.diff(expr, x)
                elif op_type == "integrate": res = sympy.integrate(expr, x)
                elif op_type == "limit": res = sympy.limit(expr, x, 0)
                elif op_type == "series": res = sympy.series(expr, x, 0, 5)
                self.display_result(str(res))
            except Exception as e: self.display_result(f"Algebra Error: {str(e)}")
        return command

    # --- Statistics Mode ---
    def render_stats_ui(self):
        ttk.Label(self.main_container, text="Data Analysis (Comma Separated)", font=("Arial", 12, "bold")).pack(pady=10)
        self.stat_input = ttk.Entry(self.main_container, font=("Consolas", 14))
        self.stat_input.pack(fill="x", padx=20)
        
        res_frame = ttk.LabelFrame(self.main_container, text="Statistics Results")
        res_frame.pack(fill="both", expand=True, padx=20, pady=20)
        self.stat_label = ttk.Label(res_frame, text="Results will appear here...", justify="left", font=("Consolas", 11))
        self.stat_label.pack(padx=10, pady=10)
        
        ttk.Button(self.main_container, text="Calculate All", command=self.run_stats).pack(pady=10)

    def run_stats(self):
        try:
            data = [float(x.strip()) for x in self.stat_input.get().split(',') if x.strip()]
            if not data: return
            out = [
                f"Count: {len(data)}",
                f"Mean: {statistics.mean(data):.4f}",
                f"Median: {statistics.median(data)}",
                f"Mode: {statistics.mode(data)}",
                f"Std Dev: {statistics.stdev(data):.4f}" if len(data) > 1 else "Std Dev: N/A",
                f"Variance: {statistics.variance(data):.4f}" if len(data) > 1 else "Variance: N/A",
                f"Min: {min(data)}",
                f"Max: {max(data)}",
                f"Sum: {sum(data)}"
            ]
            self.stat_label.config(text="\n".join(out))
        except Exception as e: self.stat_label.config(text=f"Error: {e}")

    # --- Interactive Graphing Mode ---
    def render_graphing_ui(self):
        if not HAS_MATPLOTLIB:
            ttk.Label(self.main_container, text="Matplotlib/Numpy required for Graphing.").pack()
            return

        # Controls Header
        ctrls = ttk.Frame(self.main_container)
        ctrls.pack(fill="x", padx=10, pady=5)
        
        ttk.Label(ctrls, text="f(x) =", font=("Arial", 12, "bold")).pack(side="left")
        self.graph_expr = ttk.Entry(ctrls, font=("Consolas", 12))
        self.graph_expr.pack(side="left", fill="x", expand=True, padx=5)
        self.graph_expr.insert(0, "sin(x) * x")
        self.graph_expr.bind("<Return>", lambda e: self.plot_graph())
        
        ttk.Button(ctrls, text="Update Plot", command=self.plot_graph).pack(side="left", padx=5)

        # Range and Info Bar
        info_bar = ttk.Frame(self.main_container)
        info_bar.pack(fill="x", padx=10)
        
        ttk.Label(info_bar, text="X Range:").pack(side="left")
        self.xmin_val = ttk.Entry(info_bar, width=5); self.xmin_val.insert(0, "-10"); self.xmin_val.pack(side="left", padx=2)
        ttk.Label(info_bar, text="to").pack(side="left")
        self.xmax_val = ttk.Entry(info_bar, width=5); self.xmax_val.insert(0, "10"); self.xmax_val.pack(side="left", padx=2)
        
        self.coord_label = ttk.Label(info_bar, text="Cursor: (0, 0)", foreground="blue")
        self.coord_label.pack(side="right", padx=10)

        # Plot Area
        self.fig_container = ttk.Frame(self.main_container)
        self.fig_container.pack(fill="both", expand=True)
        
        self.plot_graph()

    def plot_graph(self):
        for w in self.fig_container.winfo_children(): w.destroy()
        try:
            expr_str = self.graph_expr.get().replace('^', '**')
            xmin = float(self.xmin_val.get())
            xmax = float(self.xmax_val.get())
            
            x_sym = symbols('x')
            safe_expr = sympy.sympify(expr_str)
            f = lambdify(x_sym, safe_expr, 'numpy')
            
            x_vals = np.linspace(xmin, xmax, 1000)
            y_vals = f(x_vals)
            
            # Matplotlib setup
            fig, self.ax = plt.subplots(figsize=(6, 5), dpi=100)
            self.ax.plot(x_vals, y_vals, color='#1e88e5', linewidth=2, label=f"y = {expr_str}")
            self.ax.axhline(0, color='black', lw=0.8)
            self.ax.axvline(0, color='black', lw=0.8)
            self.ax.grid(True, linestyle=':', alpha=0.6)
            self.ax.legend()
            
            # Embedding in Tkinter
            self.canvas = FigureCanvasTkAgg(fig, master=self.fig_container)
            self.canvas.draw()
            
            # Navigation Toolbar (Zoom, Pan, Save)
            toolbar = NavigationToolbar2Tk(self.canvas, self.fig_container)
            toolbar.update()
            
            self.canvas.get_tk_widget().pack(fill="both", expand=True)
            
            # Mouse motion event for interactivity
            self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
            
        except Exception as e:
            ttk.Label(self.fig_container, text=f"Graph Error: {e}", foreground="red", wraplength=400).pack()

    def on_mouse_move(self, event):
        if event.inaxes:
            self.coord_label.config(text=f"Cursor: ({event.xdata:.2f}, {event.ydata:.2f})")

    # --- Notes App ---
    def render_notes_ui(self):
        ttk.Label(self.main_container, text="Scratchpad & Math Notes", font=("Arial", 12, "bold")).pack()
        self.notes_area = scrolledtext.ScrolledText(self.main_container, wrap=tk.WORD, font=("Segoe UI", 12))
        self.notes_area.pack(fill="both", expand=True, padx=10, pady=10)
        self.notes_area.insert("1.0", self.notes_content)
        
        btn_bar = ttk.Frame(self.main_container)
        btn_bar.pack(fill="x")
        ttk.Button(btn_bar, text="Clear Notes", command=lambda: self.notes_area.delete("1.0", tk.END)).pack(side="right", padx=10)

    # --- AI Assistant ---
    def render_ai_ui(self):
        ttk.Label(self.main_container, text="Gemini AI Math Tutor", font=("Arial", 12, "bold")).pack()
        self.ai_chat = scrolledtext.ScrolledText(self.main_container, state='disabled', wrap=tk.WORD, 
                                                bg="#fdfdfd", font=("Segoe UI", 11))
        self.ai_chat.pack(fill="both", expand=True, padx=10, pady=5)
        
        input_frame = ttk.Frame(self.main_container)
        input_frame.pack(fill="x", padx=10, pady=10)
        
        self.ai_input = ttk.Entry(input_frame, font=("Segoe UI", 12))
        self.ai_input.pack(side="left", fill="x", expand=True)
        self.ai_input.bind("<Return>", lambda e: self.ask_ai())
        
        self.ai_send_btn = ttk.Button(input_frame, text="Ask AI", command=self.ask_ai)
        self.ai_send_btn.pack(side="right", padx=5)

    def ask_ai(self):
        query = self.ai_input.get().strip()
        if not query: return
        self.ai_update_chat(f"You: {query}\n")
        self.ai_input.delete(0, tk.END)
        self.ai_send_btn.config(state='disabled')
        threading.Thread(target=self.fetch_ai_response, args=(query,), daemon=True).start()

    def ai_update_chat(self, msg):
        self.ai_chat.config(state='normal')
        self.ai_chat.insert(tk.END, msg + "\n")
        self.ai_chat.see(tk.END)
        self.ai_chat.config(state='disabled')

    def fetch_ai_response(self, query):
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-preview-09-2025:generateContent?key={self.api_key}"
        payload = {
            "contents": [{"parts": [{"text": f"Explain this math topic: {query}"}]}],
            "systemInstruction": {"parts": [{"text": "You are a specialized math tutor. Use standard notation and keep it interactive."}]}
        }
        
        retries = 0
        delays = [1, 2, 4, 8, 16]
        while retries < 5:
            try:
                response = requests.post(url, json=payload, timeout=15)
                if response.status_code == 200:
                    data = response.json()
                    text = data['candidates'][0]['content']['parts'][0]['text']
                    self.root.after(0, lambda: self.ai_update_chat(f"AI: {text}"))
                    break
                retries += 1
                if retries < 5: threading.Event().wait(delays[retries-1])
            except:
                retries += 1
                if retries < 5: threading.Event().wait(delays[retries-1])
        
        if retries == 5: self.root.after(0, lambda: self.ai_update_chat("AI: Connection error."))
        self.root.after(0, lambda: self.ai_send_btn.config(state='normal'))

if __name__ == "__main__":
    root = tk.Tk()
    style = ttk.Style()
    if "clam" in style.theme_names(): style.theme_use("clam")
    app = HybridCalculator(root)
    root.mainloop()