import co

def main() -> co.Serial:
    root = co.Serial(image = get_image())
    root["Get Data"] = co.Exec(run, f"/conducto/data/pipeline/raw")
    return root

def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    s = 55
    tf.random.set_seed(s)
    np.random.seed(s)
    data_dir = pathlib.Path("data/mini_speech_commands")

if not data_dir.exists():
    tf.keras.utils.get_file(
       'mini_speech_commands.zip',
        origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
        extract=True)

def decode(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)

def get_label(file_path):
   parts = tf.strings.split(file_path, os.path.sep)
   return parts[-2]

def get_wf_and_label(file_path):
    label = get_label(file_path)
    audio_binary = tf.io.read_file(file_path)
    wf = decode(audio_binary)
    return wf, label

def get_spectro(wf):
    zero_padding = tf.zeros([16000] - tf.shape(wf), dtype=tf.float32)
    wf = tf.cast(wf, tf.float32)
    equal = tf.concat([wf, zero_padding], 0)
    spectro = tf.signal.stft(
        equal, frame_length=255, frame_step=128)
    spectro = tf.abs(spectro)

    return spectro

def get_spectro_and_label_id(audio, label):
    spectro = get_spectro(audio)
    spectro = tf.expand_dims(spectro, -1)
    label_id = tf.argmax(label == commands)
    return spectro, label_id

def preprocess_dataset(files, autotune, commands):
   files_ds = tf.data.Dataset.from_tensor_slices(files)
   output_ds = files_ds.map(get_wf_and_label,
                            num_parallel_calls=autotune)
   output_ds = output_ds.map(
       get_spectro_and_label_id,  num_parallel_calls=autotune)
   return output_ds

commands = np.array(tf.io.gfile.listdir(str(data_dir)))
commands = commands[commands != 'README.md']

filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
filenames = tf.random.shuffle(filenames)
train_files = filenames[:6400]
validation_files = filenames[6400: 6400 + 800]
test_files = filenames[-800:]

autotune = tf.data.AUTOTUNE

files_ds = tf.data.Dataset.from_tensor_slices(train_files)
wf_ds = files_ds.map(
    get_wf_and_label, num_parallel_calls=autotune)
spectro_ds = wf_ds.map(
    get_spectro_and_label_id, num_parallel_calls=autotune)

train_ds = preprocess_dataset(train_files, autotune, commands)
v_ds = preprocess_dataset(
    validation_files, autotune, commands)
test_ds = preprocess_dataset(test_files, autotune, commands)

batch_size = 64
train_ds = train_ds.batch(batch_size)
v_ds = v_ds.batch(batch_size)

train_ds = train_ds.cache().prefetch(autotune)
v_ds = v_ds.cache().prefetch(autotune)

for spectro, _ in spectro_ds.take(1):
    input_shape = spectro.shape

num_labels = len(commands)

norm = preprocessing.Normalization()
norm.adapt(spectro_ds.map(lambda x, _: x))

model = models.Sequential([
    l.Input(shape=input_shape),
    preprocessing.Resizing(32, 32),
    norm,
    l.Conv2D(32, 3, activation='relu'),
    l.Conv2D(64, 3, activation='relu'),
    l.MaxPooling2D(),
    l.Dropout(0.25),
    l.Flatten(),
    l.Dense(128, activation='relu'),
    l.Dropout(0.5),
    l.Dense(num_labels),
])

model.summary()

model.compile(
    optimizer=tf.keras.optimizers.Adam(),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True),
    metrics=['accuracy'],
)

EPOCHS = 10
model.fit(
    train_ds,
    validation_data=v_ds,
    epochs=EPOCHS,
    callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
)

test_audio = []
test_labels = []

for i, j in test_ds:
    test_audio.append(i.numpy())
    test_labels.append(j.numpy())

test_audio = np.array(test_audio)
test_labels = np.array(test_labels)

y_pred = np.argmax(model.predict(test_audio), axis=1)
y_true = test_labels

test_acc = sum(y_pred == y_true) / len(y_true)

print(f'Test set accuracy: {test_acc:.0%}')

def get_image():
    return co.Image(
        "python:3.8-slim",
        copy_dir=".",
        reqs_py=["conducto", "tensorflow", "keras"],
    )

if __name__ == "__main__":
    co.main(default=main)

