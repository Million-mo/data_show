
from flask import Flask, render_template, request, flash, redirect, url_for, session
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
import random

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///users.db'
db = SQLAlchemy(app)

class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)

@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return redirect(url_for('sudoku'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = generate_password_hash(request.form['password'])
        
        try:
            user = User(username=username, email=email, password=password)
            db.session.add(user)
            db.session.commit()
            flash('Registration successful! Please login.', 'success')
            return redirect(url_for('login'))
        except:
            flash('Registration failed. Username or email already exists.', 'danger')
            return redirect(url_for('login'))
    
    # GET 请求直接重定向到登录页面
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['username'] = username
            flash('Login successful!', 'success')
            return redirect(url_for('sudoku'))
        else:
            flash('Invalid username or password.', 'danger')
            return render_template('login.html', username=username)
    
    return render_template('login.html', username='')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/sudoku')
def sudoku():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('sudoku.html', username=session['username'])

# 添加管理员路由
@app.route('/admin')
def admin():
    # 检查是否登录
    if 'username' not in session:
        flash('Please login first.', 'danger')
        return redirect(url_for('login'))
    
    # 获取当前用户
    user = User.query.filter_by(username=session['username']).first()
    
    # 检查用户是否存在且是管理员
    if not user:
        flash('User not found.', 'danger')
        return redirect(url_for('login'))
    
    if not user.is_admin:
        flash('Access denied. Admin privileges required.', 'danger')
        return redirect(url_for('login'))
    
    # 获取所有用户
    try:
        users = User.query.all()
        return render_template('admin.html', users=users)
    except Exception as e:
        flash(f'Error loading admin page: {str(e)}', 'danger')
        return redirect(url_for('login'))

def init_db():
    with app.app_context():
        # 删除所有现有的表
        db.drop_all()
        print("Dropped all existing tables.")
        
        # 创建新的表
        db.create_all()
        print("Database tables created successfully!")
        
        # 检查是否已存在管理员账号
        admin = User.query.filter_by(username='admin').first()
        if not admin:
            # 创建管理员账号
            admin = User(
                username='admin',
                email='admin@example.com',
                password=generate_password_hash('admin123'),
                is_admin=True
            )
            db.session.add(admin)
            db.session.commit()
            print("Admin account created successfully!")
        else:
            print("Admin account already exists!")

if __name__ == '__main__':
    init_db()
    app.run(debug=True)

