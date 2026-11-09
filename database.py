import sqlite3

from paths import BASE_DIR

DB_PATH = BASE_DIR / "database" / "abi.db"


def get_items_by_sound(sound_id: int):
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row

    try:
        cursor = connection.cursor()

        cursor.execute("""
            SELECT
                I.Item_ID,
                I.Item_Name,
                I.Item_Image_Path,
                R.Rarity_Name,
                I.Item_Price
            FROM ITEM I
            JOIN RARITY R
                ON I.Rarity_ID = R.Rarity_ID
            JOIN ITEM_SOUND ISM
                ON I.Item_ID = ISM.Item_ID
            WHERE ISM.Sound_ID = ?
        """, (sound_id,))

        return [dict(row) for row in cursor.fetchall()]

    finally:
        connection.close()