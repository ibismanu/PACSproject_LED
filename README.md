# PACSprj_AdaLED

1) riprodurre architettura dei paper -> scegliere un esempio dal paper da riprodurre

Struttura:
classe DataGeneration
	classe astratta (abc,@dataclass)
		dati salvati in una certa struttura (perchè questa? veloce da leggere, memory efficient...)
		questa struttura dati andrà dentro a surrogate model
figli: ode solver (per uno degli esempi)
(loader) 

classe surrogate model
	classe astratta
		-DataGeneration -> deve leggere i dati
		-keras
		-funzione per il training -> anche qui fare il test sul una funzione semplice
		-funzione per il predict -> test anche qui
figli: metodo paper 1/2

2) Fare test: nel codice inserire dei check 

1° step: cercare come funziona la classe astratta e capire in che struttura vogliamo salvarci i dati. Partire dalla prima classe.

no meeting prima settimana di settembre. 
