# התקנת ממיר משחקי השחמט ל-PGN

## חלק א' – עידו, על המק (פעם אחת, וגם בכל עדכון)

1. ודאו ש-Docker Desktop פועל.
2. בתיקיית הפרויקט הריצו:
   ```
   ./docker/build.sh
   ```
   זה בונה את שתי התמונות למחשב Windows (linux/amd64) ושומר הכול בתיקייה `dist/`:
   `chesspgn-images.tar`, `docker-compose.yml`, `Start Chess PGN.bat`, `Install Chess PGN.bat`.
   הבנייה הראשונה מורידה את מודל הקריאה (כ-2GB) ולוקחת זמן.
3. בדיקה על המק (אופציונלי):
   ```
   cd docker && CHESSPGN_DIR=~/ChessPGN-test docker compose up -d
   ```
   ופתחו את http://localhost:8321. על מק עם שבב Apple התמונות רצות באמולציה, ולכן לאט יותר מאשר על מחשב Windows אמיתי.

## חלק ב' – על המחשב של אבא (פעם אחת)

1. התקינו את **Docker Desktop for Windows** (docker.com) והפעילו מחדש אם מתבקש.
2. ב-Docker Desktop: Settings → General → סמנו **Start Docker Desktop when you sign in**.
3. העתיקו את תיקיית `dist` למחשב, למשל ל-`C:\Chess PGN`.
4. לחצו פעמיים על **Install Chess PGN.bat** וחכו שיסתיים (כמה דקות).
5. צרו קיצור דרך בשולחן העבודה ל-**Start Chess PGN.bat** (קליק ימני → Send to → Desktop).

## שימוש יומיומי

- לחיצה כפולה על **Start Chess PGN** בשולחן העבודה. הדפדפן ייפתח על מסך ההמרה.
- גוררים צילומי מסך (לפי סדר העמודים) ← **המרה** ← **הורדת PGN**.
- כל המשחקים נשמרים גם בתיקייה `מסמכים\Chess PGN`, משחק בתיקייה משלו.
- אם מופיעה ההודעה "מנוע הקריאה לא פועל": פתחו את Docker Desktop, חכו דקה ונסו שוב.

## עדכון גרסה

עידו בונה מחדש (חלק א'), ומעתיקים את `chesspgn-images.tar` החדש ומריצים שוב את **Install Chess PGN.bat**.
