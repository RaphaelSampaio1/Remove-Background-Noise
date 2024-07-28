import numpy as np
import matplotlib.pyplot as plt
import librosa
import soundfile as sf
import scipy.fftpack as fft
from scipy.signal import medfilt

# Carregar o arquivo de áudio
y, sr = librosa.load("short_yt.mp3", sr=None)

# Calcular o espectro de magnitude e fase
s_full, phase = librosa.magphase(librosa.stft(y))

# Calcular a potência do ruído
noise_power = np.mean(s_full[:, :int(sr*1.0)], axis=1)

# Criar uma máscara onde o sinal é maior que o ruído
mask = s_full > noise_power[:, None]

# Converter a máscara para float
mask = mask.astype(float)

# Aplicar o filtro de mediana com um kernel size ímpar, por exemplo, 3
mask = medfilt(mask, kernel_size=3)

# Aplicar a máscara ao espectro de magnitude
s_clean = s_full * mask

# Reconstruir o sinal de áudio limpo
y_clean = librosa.istft(s_clean * phase)

# Salvar o áudio limpo
sf.write('clean_audio.wav', y_clean, sr)

# Plotar o espectro original e o espectro limpo para comparação
plt.figure(figsize=(12, 6))
plt.subplot(2, 1, 1)
librosa.display.specshow(librosa.amplitude_to_db(s_full, ref=np.max), sr=sr, y_axis='log', x_axis='time')
plt.title('Audio Original')

plt.colorbar(format='%+2.0f dB')
plt.subplot(2, 1, 2)
librosa.display.specshow(librosa.amplitude_to_db(s_clean, ref=np.max), sr=sr, y_axis='log', x_axis='time')
plt.title('Audio Limpo')
plt.colorbar(format='%+2.0f dB')
plt.tight_layout()
plt.show()
