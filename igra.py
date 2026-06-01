import random

def simuliraj_monti_hol(broj_pokusaja=100000):
    ostani_dobitak = 0
    ostani_gubitak = 0
    
    promeni_dobitak = 0
    promeni_gubitak = 0
    

    pola_pokusaja = broj_pokusaja // 2

    for _ in range(pola_pokusaja):
        kutije = [0, 1, 2]
        
        poklon = random.choice(kutije)
        
        moj_izbor = random.choice(kutije)
        
        if moj_izbor == poklon:
            ostani_dobitak += 1
        else:
            ostani_gubitak += 1

    for _ in range(pola_pokusaja):
        kutije = [0, 1, 2]
        poklon = random.choice(kutije)
        moj_izbor = random.choice(kutije)
        
        moguce_za_izbacivanje = [k for k in kutije if k != poklon and k != moj_izbor]
        izbacena_kutija = random.choice(moguce_za_izbacivanje)
        

        nova_kutija = [k for k in kutije if k != moj_izbor and k != izbacena_kutija][0]
        
        if nova_kutija == poklon:
            promeni_dobitak += 1
        else:
            promeni_gubitak += 1

    print(f"=== REZULTATI SIMULACIJE ({broj_pokusaja} ukupno pokušaja) ===\n")
    
    print(f"--- Strategija: NE MENJAM (odigrano {pola_pokusaja} puta) ---")
    print(f"Dobici: {ostani_dobitak} ({ostani_dobitak / pola_pokusaja * 100:.2f}%)")
    print(f"Gubici: {ostani_gubitak} ({ostani_gubitak / pola_pokusaja * 100:.2f}%)\n")
    
    print(f"--- Strategija: MENJAM (odigrano {pola_pokusaja} puta) ---")
    print(f"Dobici: {promeni_dobitak} ({promeni_dobitak / pola_pokusaja * 100:.2f}%)")
    print(f"Gubici: {promeni_gubitak} ({promeni_gubitak / pola_pokusaja * 100:.2f}%)")

if __name__ == "__main__":
    simuliraj_monti_hol(broj_pokusaja=10000000)