import string
from datetime import datetime

# targa, mail, data, num_verbale, cf_trasgressore, cf_avvocato
class Normalizer:
    punctuations = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~'
    letters = string.ascii_letters
    mesi_encoding = {"gennaio": "01", "febbraio": "02", "marzo": "03", "aprile": "04", "maggio": "05", "giugno": "06", "luglio": "07", "agosto": "08", "settembre": "09", "ottobre": "10", "novembre": "11", "dicembre": "12"}
    

    def normalize(self, entity, label):
        if label.lower() == "num_verbale":
            return self.num_verbale(entity)
        elif label.lower() == "targa":
            return self.targa(entity)
        elif label.lower() == "mail":
            return self.mail(entity)
        elif label.lower() == "data":
            return self.data(entity)
        elif label.lower() == "cf_trasgressore":
            return self.cf_trasgressore(entity)
        elif label.lower() == "cf_avvocato":
            return self.cf_avvocato(entity)
        elif label.lower() == "destinatario":
            return self.destinatario(entity)
        else:
            return entity
            
    def targa(self, entity):
        entity = self.remove_punc_beginning_end(entity)
        entity = entity.replace(" ", "")
        return entity

            
    def mail(self, entity):
        entity = self.remove_punc_beginning_end(entity)
        entity = entity.replace(" ", "")
        return entity


    def data(self, entity):
        try:
            day, month, year = None, None, None
            entity = self.remove_punc_beginning_end(entity)
            
            #whitespaces
            if any(char in string.ascii_letters for char in entity):
                found_mese = False
                for mese in self.mesi_encoding.keys():
                    if mese in entity:
                        found_mese = True
                        day, month, year = entity.split(" ")
                        month = self.mesi_encoding[month]
                if not found_mese:
                    return entity
            else:
                entity = entity.replace(" ", "")
             
            if not day:    
                if "." in entity:
                    day, month, year = entity.split(".")
                elif "/" in entity:
                    day, month, year = entity.split("/")
            if len(year) == 2:
                year = "20" + year

            date = datetime( month=int(month), day=int(day), year=int(year))
            entity = date.strftime("%d/%m/%Y")
            return entity
        except:                # if there is any error return the same entity
            return entity

    def num_verbale(self, entity):
        entity = self.remove_punc_beginning_end(entity)
        entity = entity.replace(" ", "")
        return entity


    def destinatario(self, entity):
        entity = self.remove_punc_beginning_end(entity)
        entity = entity.lower()
        return entity


    def cf_trasgressore(self, entity):
        entity = self.remove_punc_beginning_end(entity)
        entity = entity.replace(" ", "")
        return entity


    def cf_avvocato(self, entity):
        entity = self.remove_punc_beginning_end(entity)
        entity = entity.replace(" ", "")
        return entity


    def remove_punc_beginning_end(self, entity):
        try:
            if entity[0] in self.punctuations:
                entity = entity[1:]
            if entity[-1] in self.punctuations:
                entity = entity[:-1]
            return entity
        except:
                return entity
    
    
normalizer = Normalizer()
data = normalizer.data("21 agosto 1992")
print(data)