import csv

class CSVLogger:
    def __init__(self, filename="training_log.csv"):
        self.filename = filename
        self.file = open(filename, mode='a', newline='')
        self.writer = None

    def log(self, step, metrics):
        row = {"step": step, **metrics}
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=row.keys())
            if self.file.tell() == 0: # Écrit l'en-tête si fichier vide
                self.writer.writeheader()
        
        self.writer.writerow(row)

    def flush(self):
        self.file.flush()

    def close(self):
        self.file.close()