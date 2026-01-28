#!/usr/bin/env python3
import sys
import os
import json
import time
import wave
import torch
import pyaudio
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QPushButton, QLabel, QLineEdit, 
                             QStackedWidget, QFileDialog, QTextEdit, QComboBox,
                             QMessageBox, QFrame, QRadioButton, QButtonGroup, QSpacerItem,
                             QDialog, QGridLayout)
from PyQt5.QtCore import Qt, pyqtSignal, QThread, QBuffer, QByteArray
from PyQt5.QtGui import QFont, QMovie, QPalette, QColor
from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq, pipeline

# Thread for audio recording
torch_device = "cuda" if torch.cuda.is_available() else "cpu"

def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)

class RecordingThread(QThread):
    update_status = pyqtSignal(str)
    def __init__(self, filename):
        super().__init__()
        self.filename = filename
        self.is_recording = False
        self.is_paused = False

    def run(self):
        self.is_recording = True
        self.update_status.emit("Recording started...")
        CHUNK = 1024
        FORMAT = pyaudio.paInt16
        CHANNELS = 1
        RATE = 44100
        p = pyaudio.PyAudio()
        stream = p.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)
        frames = []
        while self.is_recording:
            if not self.is_paused:
                data = stream.read(CHUNK)
                frames.append(data)
            else:
                time.sleep(0.1)
        stream.stop_stream()
        stream.close()
        p.terminate()
        ensure_dir(self.filename)
        wf = wave.open(self.filename, 'wb')
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(p.get_sample_size(FORMAT))
        wf.setframerate(RATE)
        wf.writeframes(b''.join(frames))
        wf.close()
        self.update_status.emit(f"Recording saved to {self.filename}")

    def pause(self):
        self.is_paused = True
        self.update_status.emit("Recording paused")

    def resume(self):
        self.is_paused = False
        self.update_status.emit("Recording resumed")

    def stop(self):
        self.is_recording = False
        self.update_status.emit("Recording stopped")

# Thread for transcription
def load_whisper_pipeline():
    model_id = "openai/whisper-large-v3-turbo"
    proc = AutoProcessor.from_pretrained(model_id)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_id, 
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    model.to(torch_device)
    return pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=proc.tokenizer,
        feature_extractor=proc.feature_extractor,
        device=0 if torch.cuda.is_available() else -1
    )

whisper_pipe = None
class TranscriptionThread(QThread):
    transcription_complete = pyqtSignal(str)
    status_update = pyqtSignal(str)

    def __init__(self, audio_file):
        super().__init__()
        self.audio_file = audio_file

    def run(self):
        global whisper_pipe
        try:
            self.status_update.emit("Loading model...")
            if whisper_pipe is None:
                whisper_pipe = load_whisper_pipeline()
            self.status_update.emit("Transcribing...")
            result = whisper_pipe(
                self.audio_file, 
                generate_kwargs={"language": "en", "task": "transcribe"}
            )
            self.transcription_complete.emit(result.get("text", ""))
        except Exception as e:
            self.transcription_complete.emit(f"Error during transcription: {str(e)}")

# Chat dialog for unclear audio
class ChatDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Communication Assistant")
        self.setFixedSize(500, 400)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        
        # Title
        title = QLabel("Audio not clear. Please select your need:")
        title.setFont(QFont("Ubuntu", 14, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Common needs for dysarthria patients
        needs_layout = QGridLayout()
        
        needs = [
            ("🍽️ Food", "I need food"),
            ("💧 Water", "I need water"),
            ("🚽 Washroom", "I need to use the washroom"),
            ("👋 Greeting", "Hello, how are you?"),
            ("🏥 Help", "I need help"),
            ("😴 Rest", "I need to rest")
        ]
        
        row, col = 0, 0
        for icon_text, message in needs:
            btn = QPushButton(icon_text)
            btn.setFixedSize(150, 80)
            btn.setStyleSheet("""
                QPushButton {
                    background-color: #e74c3c;
                    color: white;
                    border-radius: 10px;
                    font-size: 12px;
                    font-weight: bold;
                }
                QPushButton:hover {
                    background-color: #c0392b;
                }
            """)
            btn.clicked.connect(lambda checked, msg=message: self.select_need(msg))
            needs_layout.addWidget(btn, row, col)
            
            col += 1
            if col > 1:
                col = 0
                row += 1
        
        layout.addLayout(needs_layout)
        
        # Selected message display
        self.selected_label = QLabel("Selected: None")
        self.selected_label.setFont(QFont("Ubuntu", 12))
        self.selected_label.setStyleSheet("border: 1px solid #bdc3c7; padding: 10px; border-radius: 5px;")
        layout.addWidget(self.selected_label)
        
        # Close button
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        layout.addWidget(close_btn)

    def select_need(self, message):
        self.selected_label.setText(f"Selected: {message}")

# User database for registration/login
class UserDatabase:
    def __init__(self, database_file="users.json"):
        self.database_file = database_file
        self.users = self.load_database()

    def load_database(self):
        try:
            with open(self.database_file, 'r') as f:
                return json.load(f)
        except:
            return {}

    def save_database(self):
        with open(self.database_file, 'w') as f:
            json.dump(self.users, f, indent=4)

    def register_user(self, username, password, name, age, gender, symptoms):
        if username in self.users:
            return False
        self.users[username] = {"password": password, "name": name, "age": age, "gender": gender, "symptoms": symptoms}
        self.save_database()
        return True

    def authenticate(self, username, password):
        return username in self.users and self.users[username]["password"] == password

    def get_user_data(self, username):
        return self.users.get(username)

# Base style sheet for buttons and frames
GLOBAL_STYLES = """
QPushButton {
    background-color: #3498db;
    color: white;
    border-radius: 8px;
    padding: 8px 16px;
    font-size: 14px;
}
QPushButton:hover {
    background-color: #2980b9;
}
QFrame { 
    border-radius: 12px;
}
QLineEdit, QTextEdit {
    border: 1px solid #bdc3c7;
    border-radius: 6px;
    padding: 6px;
}
"""

# Login page UI
class LoginPage(QWidget):
    login_successful = pyqtSignal(str)
    show_register = pyqtSignal()

    def __init__(self, user_db):
        super().__init__()
        self.user_db = user_db
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(GLOBAL_STYLES)
        layout = QVBoxLayout(self)
        layout.addItem(QSpacerItem(20, 40))
        lbl = QLabel("Slurred Speech transcriber", alignment=Qt.AlignCenter)
        lbl.setFont(QFont("Ubuntu", 24, QFont.Bold))
        lbl.setStyleSheet("color: #2c3e50; margin-bottom: 20px;")
        layout.addWidget(lbl)
        frame = QFrame()
        frame.setAutoFillBackground(True)
        pal = frame.palette()
        pal.setColor(QPalette.Window, QColor('#ecf0f1'))
        frame.setPalette(pal)
        frame.setStyleSheet("QFrame { background-color: #ecf0f1; }")
        fl = QVBoxLayout(frame)
        fl.setContentsMargins(30,30,30,30)
        title = QLabel("Login", alignment=Qt.AlignCenter)
        title.setFont(QFont("Ubuntu", 16))
        fl.addWidget(title)
        self.user_in = QLineEdit(); self.user_in.setPlaceholderText("Username"); fl.addWidget(self.user_in)
        self.pass_in = QLineEdit(); self.pass_in.setPlaceholderText("Password"); self.pass_in.setEchoMode(QLineEdit.Password); fl.addWidget(self.pass_in)
        btn = QPushButton("Login"); btn.clicked.connect(self.login); fl.addWidget(btn)
        reg = QPushButton("Don't have an account? Register"); reg.clicked.connect(self.show_register.emit); fl.addWidget(reg)
        layout.addWidget(frame)
        layout.addItem(QSpacerItem(20, 40))

    def login(self):
        u = self.user_in.text().strip(); p = self.pass_in.text().strip()
        if not u or not p:
            QMessageBox.warning(self, "Error", "Enter username and password.")
        elif self.user_db.authenticate(u, p):
            self.login_successful.emit(u)
        else:
            QMessageBox.warning(self, "Error", "Invalid credentials.")

# Registration page UI
class RegisterPage(QWidget):
    register_successful = pyqtSignal()
    show_login = pyqtSignal()

    def __init__(self, user_db):
        super().__init__()
        self.user_db = user_db
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(GLOBAL_STYLES)
        layout = QVBoxLayout(self)
        lbl = QLabel("Speech Transcription App", alignment=Qt.AlignCenter)
        lbl.setFont(QFont("Ubuntu", 24, QFont.Bold))
        lbl.setStyleSheet("color: #2c3e50; margin-bottom:20px;")
        layout.addWidget(lbl)
        frame = QFrame()
        frame.setAutoFillBackground(True)
        pal = frame.palette()
        pal.setColor(QPalette.Window, QColor('#ecf0f1'))
        frame.setPalette(pal)
        frame.setStyleSheet("QFrame { background-color: #ecf0f1; }")
        fl = QVBoxLayout(frame); fl.setContentsMargins(30,30,30,30)
        title = QLabel("Register", alignment=Qt.AlignCenter); title.setFont(QFont("Ubuntu",16)); fl.addWidget(title)
        self.user_in = QLineEdit(); self.user_in.setPlaceholderText("Username"); fl.addWidget(self.user_in)
        self.pass_in = QLineEdit(); self.pass_in.setPlaceholderText("Password"); self.pass_in.setEchoMode(QLineEdit.Password); fl.addWidget(self.pass_in)
        self.name_in = QLineEdit(); self.name_in.setPlaceholderText("Full Name"); fl.addWidget(self.name_in)
        self.age_in  = QLineEdit(); self.age_in.setPlaceholderText("Age"); fl.addWidget(self.age_in)
        gbox = QHBoxLayout(); self.gender_group = QButtonGroup(self)
        for txt in ["Male","Female","Other"]:
            rb = QRadioButton(txt); self.gender_group.addButton(rb); gbox.addWidget(rb)
        fl.addLayout(gbox)
        self.symp_in = QTextEdit(); self.symp_in.setPlaceholderText("Symptoms (optional)"); fl.addWidget(self.symp_in)
        btn = QPushButton("Register"); btn.clicked.connect(self.register); fl.addWidget(btn)
        login = QPushButton("Already have account? Login"); login.clicked.connect(self.show_login.emit); fl.addWidget(login)
        layout.addWidget(frame)

    def register(self):
        u = self.user_in.text().strip(); p = self.pass_in.text().strip(); nm = self.name_in.text().strip(); ag = self.age_in.text().strip()
        gen = next((b.text() for b in self.gender_group.buttons() if b.isChecked()), None)
        sym = self.symp_in.toPlainText().strip()
        if not all([u,p,nm,ag,gen]):
            QMessageBox.warning(self, "Error", "Fill all fields.")
            return
        if not ag.isdigit():
            QMessageBox.warning(self, "Error", "Age must be a number.")
            return
        if self.user_db.register_user(u,p,nm,ag,gen,sym):
            QMessageBox.information(self, "Success", "Registered! Please login.")
            self.show_login.emit()
        else:
            QMessageBox.warning(self, "Error", "Username exists.")

# Main application page
class MainPage(QWidget):
    logout_signal = pyqtSignal()

    def __init__(self, username, user_data):
        super().__init__()
        self.username = username
        self.user_data = user_data
        self.recording_thread = None
        self.transcription_thread = None
        self.current_recording_path = None
        # Two different recording paths
        self.clear_audio_path = os.path.join("recordings", "clear_audio", f"{username}.wav")
        self.unclear_audio_path = os.path.join("recordings", "unclear_audio", f"{username}.wav")
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(GLOBAL_STYLES)
        layout = QVBoxLayout(self)
        # Header
        hl = QHBoxLayout(); lbl = QLabel(f"Welcome, {self.user_data['name']}"); lbl.setFont(QFont("Ubuntu",16,QFont.Bold)); hl.addWidget(lbl)
        btn = QPushButton("Logout"); btn.clicked.connect(self.logout_signal.emit); hl.addStretch(); hl.addWidget(btn); layout.addLayout(hl)
        # Content
        cl = QHBoxLayout()
        rf = QFrame()
        rf.setAutoFillBackground(True)
        pal_rf = rf.palette(); pal_rf.setColor(QPalette.Window, QColor('#ffffff'))
        rf.setPalette(pal_rf)
        rf.setStyleSheet("QFrame{background-color:#ffffff; border-radius:8px;}")
        rfl=QVBoxLayout(rf)
        rfl.addWidget(QLabel("Audio Recording", alignment=Qt.AlignCenter))
        
        # Two record buttons
        record_buttons_layout = QHBoxLayout()
        self.record_clear_btn = QPushButton("Record Audio")
        self.record_unclear_btn = QPushButton("Clear")
        self.record_clear_btn.setStyleSheet("QPushButton { background-color: #27ae60; }")
        self.record_unclear_btn.setStyleSheet("QPushButton { background-color: #e74c3c; }")
        self.record_clear_btn.clicked.connect(self.start_clear_recording)
        self.record_unclear_btn.clicked.connect(self.start_unclear_recording)
        record_buttons_layout.addWidget(self.record_clear_btn)
        record_buttons_layout.addWidget(self.record_unclear_btn)
        rfl.addLayout(record_buttons_layout)
        
        # Control buttons
        ctrl = QHBoxLayout()
        self.pause_btn, self.resume_btn, self.stop_btn = [QPushButton(txt) for txt in ("Pause","Resume","Stop")]
        self.pause_btn.clicked.connect(self.pause_recording)
        self.resume_btn.clicked.connect(self.resume_recording)
        self.stop_btn.clicked.connect(self.stop_recording)
        for b in [self.pause_btn, self.resume_btn, self.stop_btn]: ctrl.addWidget(b)
        self.pause_btn.setEnabled(False); self.resume_btn.setEnabled(False); self.stop_btn.setEnabled(False)
        rfl.addLayout(ctrl)
        
        # File upload section
        upl = QVBoxLayout(); upl_frame = QFrame(); upl_frame.setLayout(upl)
        upl_frame.setAutoFillBackground(True)
        pal_u = upl_frame.palette(); pal_u.setColor(QPalette.Window,QColor('#ffffff')); upl_frame.setPalette(pal_u)
        upl_frame.setStyleSheet("QFrame{background-color:#ffffff; border-radius:8px;}")
        upl.addWidget(QLabel("Or upload audio file:", alignment=Qt.AlignLeft))
        upb = QPushButton("Browse Files..."); upb.clicked.connect(self.browse_files); upl.addWidget(upb)
        self.selected_file_label = QLabel("No file selected"); upl.addWidget(self.selected_file_label)
        rfl.addWidget(upl_frame)
        
        self.status_label = QLabel("Ready"); rfl.addWidget(self.status_label)
        cl.addWidget(rf)
        
        # Transcription section
        tf = QFrame()
        tf.setAutoFillBackground(True)
        pal_tf = tf.palette(); pal_tf.setColor(QPalette.Window, QColor('#ffffff'))
        tf.setPalette(pal_tf)
        tf.setStyleSheet("QFrame{background-color:#ffffff; border-radius:8px;}")
        tfl=QVBoxLayout(tf)
        tfl.addWidget(QLabel("Transcription", alignment=Qt.AlignCenter))
        self.trans_text = QTextEdit(); self.trans_text.setReadOnly(True); self.trans_text.setPlaceholderText("Transcription will appear here"); self.trans_text.setMinimumHeight(300); font = QFont("Ubuntu", 18)
        self.trans_text.setFont(font)
        tfl.addWidget(self.trans_text)
        self.trans_btn = QPushButton("Transcribe Audio"); self.trans_btn.setEnabled(False); self.trans_btn.clicked.connect(self.transcribe_audio); tfl.addWidget(self.trans_btn)
        cl.addWidget(tf)
        layout.addLayout(cl)

    def start_clear_recording(self):
        self.current_recording_path = self.clear_audio_path
        self.start_recording()

    def start_unclear_recording(self):
        self.current_recording_path = self.unclear_audio_path
        self.start_recording()

    def start_recording(self):
        ensure_dir(self.current_recording_path)
        self.recording_thread = RecordingThread(self.current_recording_path)
        self.recording_thread.update_status.connect(self.update_status)
        self.recording_thread.start()
        self.record_clear_btn.setEnabled(False)
        self.record_unclear_btn.setEnabled(False)
        self.pause_btn.setEnabled(True)
        self.stop_btn.setEnabled(True)

    def pause_recording(self):
        if self.recording_thread:
            self.recording_thread.pause()
            self.pause_btn.setEnabled(False); self.resume_btn.setEnabled(True)

    def resume_recording(self):
        if self.recording_thread:
            self.recording_thread.resume()
            self.pause_btn.setEnabled(True); self.resume_btn.setEnabled(False)

    def stop_recording(self):
        if self.recording_thread:
            self.recording_thread.stop()
            self.recording_thread.wait()
            self.recording_thread = None
            self.record_clear_btn.setEnabled(True)
            self.record_unclear_btn.setEnabled(True)
            self.pause_btn.setEnabled(False); self.resume_btn.setEnabled(False); self.stop_btn.setEnabled(False)
            self.trans_btn.setEnabled(True)

    def browse_files(self):
        file, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "", "Audio Files (*.wav *.mp3)")
        if file:
            self.current_recording_path = file
            self.selected_file_label.setText(os.path.basename(file))
            self.trans_btn.setEnabled(True)

    def update_status(self, msg):
        self.status_label.setText(msg)

    def transcribe_audio(self):
        if not self.current_recording_path:
            QMessageBox.warning(self, "Error", "No audio file selected.")
            return
        
        # Check if audio is from unclear folder
        if "unclear_audio" in self.current_recording_path:
            QMessageBox.information(self, "Audio Not Clear", "Audio not clear. Opening communication assistant...")
            chat_dialog = ChatDialog(self)
            chat_dialog.exec_()
            return
        
        # Proceed with transcription for clear audio
        self.trans_btn.setEnabled(False)
        self.transcription_thread = TranscriptionThread(self.current_recording_path)
        self.transcription_thread.status_update.connect(self.update_status)
        self.transcription_thread.transcription_complete.connect(self.on_transcription_complete)
        self.transcription_thread.start()

    def on_transcription_complete(self, text):
        self.trans_text.setText(text)
        self.trans_btn.setEnabled(True)

# Main window to manage pages
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Speech Transcription App")
        self.resize(800, 600)
        self.stack = QStackedWidget()
        self.user_db = UserDatabase()
        self.login_page = LoginPage(self.user_db)
        self.register_page = RegisterPage(self.user_db)
        self.login_page.login_successful.connect(self.on_login_success)
        self.login_page.show_register.connect(lambda: self.stack.setCurrentWidget(self.register_page))
        self.register_page.show_login.connect(lambda: self.stack.setCurrentWidget(self.login_page))
        self.stack.addWidget(self.login_page)
        self.stack.addWidget(self.register_page)
        self.setCentralWidget(self.stack)

    def on_login_success(self, username):
        user_data = self.user_db.get_user_data(username)
        self.main_page = MainPage(username, user_data)
        self.main_page.logout_signal.connect(lambda: self.stack.setCurrentWidget(self.login_page))
        self.stack.addWidget(self.main_page)
        self.stack.setCurrentWidget(self.main_page)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
