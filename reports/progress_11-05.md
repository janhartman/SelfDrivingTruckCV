# Poročilo o napredku - teden 8.5. - 12.5.
## Jan Hartman

Število porabljenih ur: cca. 15

Dejavnosti:  
* Zajemanje screenshotov
* Implementacija procesiranja slik (screenshotov): 
  - Canny edge detector
  - Region of interest (samo cesta)
  - Probabilistic Hough transform
  - Iskanje robov ceste - tu moram še delati na robustnosti, potrebno je malo bolj zanesljivo
* Izrisovanje procesiranih slik z narisanimi robovi ceste
* Testno pošiljanje signalov igri (uporabljen DirectKeys, ker samo pošiljanje pritiskov gumbov (PyAutoGUI) ne deluje)

Trenutno stanje: Za igro sem izbral Euro Truck Simulator 2. 
Osnovno procesiranje slik in enostavno zaznavanje robov ceste deluje (če teče v zanki in sproti izrisuje rezultat, dobimo okoli 10-15 FPS).

Naslednji koraki:  
* izboljšanje zaznavanja robov (kakšen nasvet bi prišel prav)
* določanje potrebne smeri glede na robove (osnovni if-then)
* (če bo čas) ustvarjanje učnih podatkov in implementacija konkretne AI namesto if stavkov