import psycopg
from psycopg.rows import dict_row

# Podaci za povezivanje na Postgres unutar Dockera
DB_PARAMS = {
    "host": "localhost",
    "port": 5432,
    "user": "moj_user",
    "password": "moja_sifra",
    "dbname": "moji_filmovi"
}

def main():
    with psycopg.connect(**DB_PARAMS, row_factory=dict_row) as conn:
        with conn.cursor() as cur:
            
            print("1. Pripremam bazu i ekstenziju...")

            cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
            

            cur.execute("""
                CREATE TABLE IF NOT EXISTS filmovi_tabela (
                    id SERIAL PRIMARY KEY,
                    naslov TEXT,
                    zanr TEXT,
                    embedding vector(4)
                );
            """)
            conn.commit()

            cur.execute("TRUNCATE TABLE filmovi_tabela;")
            conn.commit()

            print("2. Ubacujem podatke preko standardnog SQL-a...")
            filmovi = [
                ("The Matrix", "Sci-Fi", "[0.9, 0.1, 0.0, 0.0]"),
                ("Inception", "Sci-Fi", "[0.8, 0.05, 0.1, 0.1]"),
                ("The Notebook", "Romance", "[0.0, 0.0, 0.8, 0.9]")
            ]

            for naslov, zanr, vektor_str in filmovi:
                cur.execute("""
                    INSERT INTO filmovi_tabela (naslov, zanr, embedding) 
                    VALUES (%s, %s, %s);
                """, (naslov, zanr, vektor_str))
            conn.commit()
            print(f" -> Uspešno ubačeno {len(filmovi)} filma.")

            print("\n3. Izvršavam SQL vektorsku pretragu (Kosinusna udaljenost)...")
            upit_vektor = "[0.9, 0.2, 0.0, 0.0]"
            

            cur.execute("""
                SELECT naslov, zanr, (embedding <=> %s) AS distanca 
                FROM filmovi_tabela 
                ORDER BY embedding <=> %s 
                LIMIT 2;
            """, (upit_vektor, upit_vektor))
            
            for red in cur.fetchall():
                slicnost = 1 - red['distanca']
                print(f" -> Film: {red['naslov']} | Sličnost: {slicnost:.4f}")

            print("\n4. Izvršavam pretragu uz standardno SQL filtriranje WHERE...")
            cur.execute("""
                SELECT naslov, zanr, (embedding <=> %s) AS distanca 
                FROM filmovi_tabela 
                WHERE zanr = %s
                ORDER BY embedding <=> %s 
                LIMIT 2;
            """, (upit_vektor, "Romance", upit_vektor))

            for red in cur.fetchall():
                slicnost = 1 - red['distanca']
                print(f" -> [{red['zanr']}] Film: {red['naslov']} | Sličnost: {slicnost:.4f}")

if __name__ == "__main__":
    main()