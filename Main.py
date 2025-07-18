import asyncio
import re
import json
import os
import shutil
from datetime import datetime
from telethon import TelegramClient, events
from telethon.tl.functions.messages import GetHistoryRequest
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
import numpy as np

# --- Telegram Configuration ---
api_id = 9547079
api_hash = "b00071b9704865c62d30126659005ba0"
bot_token = "7828074748:AAEBpXZxbZ7Z0p9HEzCdcxRcwD2yhUBqGog"

# Group IDs
source_group_id = -1002161380864  # Group to read results from
target_group_id = -1002804695717  # Group to send predictions to

# --- File Storage ---
data_file = "lich_su_phien.json"
backup_file = "lich_su_phien_backup.json"
log_file = "log_du_doan.txt"

# --- Regex Pattern for New Message Format ---
pattern = re.compile(
    r"(?:\[Room\s*\].*?)?KQ\s*KỲ\s*#(\d+).*?([\d\s]+)\s*(TÀI|XỈU)\s*\((\d+)\)",
    re.IGNORECASE | re.UNICODE)

# --- Initialize Clients ---
client = TelegramClient('session_ai', api_id, api_hash)
bot = TelegramClient('bot', api_id, api_hash).start(bot_token=bot_token)

# --- Logging ---
def log(text):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_entry = f"[{timestamp}] {text}\n"
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(log_entry)
    print(log_entry.strip())

# --- Data Management ---
def load_results():
    try:
        if os.path.exists(data_file):
            with open(data_file, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data if isinstance(data, list) else []
        return []
    except (FileNotFoundError, json.JSONDecodeError, Exception) as e:
        log(f"Error loading data: {str(e)}")
        if os.path.exists(backup_file):
            try:
                with open(backup_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
            except:
                pass
        return []

def save_results(results):
    try:
        if os.path.exists(data_file):
            shutil.copyfile(data_file, backup_file)
            
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=4, ensure_ascii=False)
    except Exception as e:
        log(f"Error saving data: {str(e)}")

def validate_data(data):
    """Validate and clean data"""
    if not isinstance(data, list):
        return []
    
    valid_data = []
    seen_kys = set()
    
    for entry in data:
        try:
            if not all(key in entry for key in ["ky", "dice", "sum", "tai_xiu", "chan_le"]):
                continue
                
            if entry["ky"] in seen_kys:
                continue
                
            seen_kys.add(entry["ky"])
            valid_data.append(entry)
        except:
            continue
    
    valid_data.sort(key=lambda x: x["ky"])
    return valid_data

# --- Feature Engineering ---
def extract_features(data):
    X, y_tx, y_cl = [], [], []
    for i in range(6, len(data) - 1):
        feature = []
        current = data[i]

        # Dice values and sum
        feature += current["dice"]
        feature.append(current["sum"])

        # Recent trends
        last_tx = [1 if d["tai_xiu"] == "TÀI" else 0 for d in data[i-6:i]]
        last_cl = [1 if d["chan_le"] == "CHẴN" else 0 for d in data[i-6:i]]

        # Consecutive patterns
        for n in [2, 3, 4, 5]:
            feature.append(int(all(x == last_tx[-1] for x in last_tx[-n:])))
            feature.append(int(all(x == last_cl[-1] for x in last_cl[-n:])))

        # Alternating patterns
        def is_alt(seq):
            return int(all(seq[j] != seq[j+1] for j in range(len(seq)-1)))
        
        feature.append(is_alt(last_tx[-4:]))
        feature.append(is_alt(last_cl[-4:]))

        # Common patterns
        def match_pattern(seq, pattern):
            return int(seq[-len(pattern):] == pattern)
        
        patterns = [
            [1, 1], [0, 0], [1, 0], [0, 1],
            [1, 1, 1], [0, 0, 0]
        ]
        
        for p in patterns:
            feature.append(match_pattern(last_tx, p))
            feature.append(match_pattern(last_cl, p))

        # Basic features
        feature.append(int(current["sum"] > 10))  # Tai/Xiu threshold
        feature.append(int(current["sum"] % 2 == 0))  # Chan/Le

        X.append(feature)
        y_tx.append(1 if data[i+1]["tai_xiu"] == "TÀI" else 0)
        y_cl.append(1 if data[i+1]["chan_le"] == "CHẴN" else 0)
    
    return np.array(X), np.array(y_tx), np.array(y_cl)

# --- Model Training ---
def train_models(data):
    try:
        X, y_tx, y_cl = extract_features(data)

        model_tx = VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("gb", GradientBoostingClassifier(n_estimators=100))
        ], voting='soft')

        model_cl = VotingClassifier(estimators=[
            ("rf", RandomForestClassifier(n_estimators=100)),
            ("gb", GradientBoostingClassifier(n_estimators=100))
        ], voting='soft')

        model_tx.fit(X, y_tx)
        model_cl.fit(X, y_cl)

        return model_tx, model_cl
    except Exception as e:
        log(f"Training error: {str(e)}")
        return None, None

# --- Prediction ---
def predict_next(models, data):
    try:
        if len(data) < 7:
            return None
            
        current = data[-1]
        i = len(data) - 1

        feature = []
        feature += current["dice"]
        feature.append(current["sum"])

        last_tx = [1 if d["tai_xiu"] == "TÀI" else 0 for d in data[i-6:i]]
        last_cl = [1 if d["chan_le"] == "CHẴN" else 0 for d in data[i-6:i]]

        # Same feature extraction as in training
        for n in [2, 3, 4, 5]:
            feature.append(int(all(x == last_tx[-1] for x in last_tx[-n:])))
            feature.append(int(all(x == last_cl[-1] for x in last_cl[-n:])))

        def is_alt(seq):
            return int(all(seq[j] != seq[j+1] for j in range(len(seq)-1)))
        
        feature.append(is_alt(last_tx[-4:]))
        feature.append(is_alt(last_cl[-4:]))

        patterns = [
            [1, 1], [0, 0], [1, 0], [0, 1],
            [1, 1, 1], [0, 0, 0]
        ]
        
        for p in patterns:
            feature.append(int(last_tx[-len(p):] == p))
            feature.append(int(last_cl[-len(p):] == p))

        feature.append(int(current["sum"] > 10))
        feature.append(int(current["sum"] % 2 == 0))

        model_tx, model_cl = models
        if model_tx is None or model_cl is None:
            return None

        pred_tx = model_tx.predict([feature])[0]
        prob_tx = model_tx.predict_proba([feature])[0][pred_tx]
        pred_cl = model_cl.predict([feature])[0]
        prob_cl = model_cl.predict_proba([feature])[0][pred_cl]
        
        return {
            "tai_xiu": "TÀI" if pred_tx == 1 else "XỈU",
            "chan_le": "CHẴN" if pred_cl == 1 else "LẺ",
            "prob_tx": round(prob_tx * 100, 2),
            "prob_cl": round(prob_cl * 100, 2)
        }
    except Exception as e:
        log(f"Prediction error: {str(e)}")
        return None

# --- Group Scanning ---
async def scan_group():
    try:
        if not client.is_connected():
            await client.connect()
            
        group = await client.get_entity(source_group_id)
        all_results = load_results()
        existing_kys = {d["ky"] for d in all_results}
        
        offset_id = 0
        limit = 100
        total_added = 0
        request_count = 0
        
        while True:
            try:
                history = await client(GetHistoryRequest(
                    peer=group,
                    limit=limit,
                    offset_id=offset_id,
                    max_id=0,
                    min_id=0,
                    add_offset=0,
                    hash=0
                ))
                
                if not history.messages:
                    break
                    
                new_results = []
                for msg in history.messages:
                    if msg.message:
                        match = pattern.search(msg.message.upper())
                        if match:
                            ky = int(match.group(1))
                            if ky not in existing_kys:
                                dice = list(map(int, match.group(2).strip().split()))
                                tai_xiu = match.group(3).upper()
                                total = int(match.group(4))
                                
                                # Validate data
                                if sum(dice) != total or tai_xiu not in ["TÀI", "XỈU"]:
                                    continue
                                    
                                chan_le = "CHẴN" if total % 2 == 0 else "LẺ"
                                
                                new_results.append({
                                    "ky": ky,
                                    "dice": dice,
                                    "sum": total,
                                    "tai_xiu": tai_xiu,
                                    "chan_le": chan_le,
                                    "timestamp": msg.date.isoformat() if msg.date else None
                                })
                                existing_kys.add(ky)
                
                if new_results:
                    all_results.extend(new_results)
                    total_added += len(new_results)
                    
                if len(history.messages) < limit:
                    break
                    
                offset_id = history.messages[-1].id
                request_count += 1
                
                # Add delay to avoid rate limiting
                if request_count % 5 == 0:
                    await asyncio.sleep(2)
                    
            except Exception as e:
                log(f"History scan error: {str(e)}")
                break
                
        all_results.sort(key=lambda x: x["ky"])
        
        if total_added > 0:
            save_results(all_results)
            log(f"Added {total_added} new rounds. Total: {len(all_results)}")
            
        return all_results
        
    except Exception as e:
        log(f"Group scan failed: {str(e)}")
        return []

# --- Message Handler ---
@client.on(events.NewMessage(chats=source_group_id))
async def handle_new_message(event):
    try:
        msg = event.message.message
        match = pattern.search(msg.upper())
        if not match:
            return

        ky = int(match.group(1))
        dice = list(map(int, match.group(2).strip().split()))
        tai_xiu = match.group(3).upper()
        total = int(match.group(4))
        
        # Data validation
        if sum(dice) != total:
            log(f"Sum mismatch in round {ky}: {dice} vs {total}")
            return
            
        if tai_xiu not in ["TÀI", "XỈU"]:
            log(f"Invalid Tai/Xiu value in round {ky}: {tai_xiu}")
            return

        data = load_results()
        
        # Check for duplicates
        if any(d["ky"] == ky for d in data):
            log(f"Round {ky} already exists")
            return
            
        # Determine Chan/Le
        chan_le = "CHẴN" if total % 2 == 0 else "LẺ"
        
        new_entry = {
            "ky": ky,
            "dice": dice,
            "sum": total,
            "tai_xiu": tai_xiu,
            "chan_le": chan_le,
            "timestamp": datetime.now().isoformat()
        }
        
        data.append(new_entry)
        save_results(data)
        log(f"Added round {ky}: {dice} - {tai_xiu}/{chan_le}")

        if len(data) > 20:
            models = train_models(data)
            if models[0] is None or models[1] is None:
                return
                
            prediction = predict_next(models, data)
            if prediction is None:
                return

            text = (
                f"🏆 DỰ ĐOÁN KỲ {ky + 1} 🏆\n"
                f"🎲 Tài/Xỉu: {prediction['tai_xiu']} ({prediction['prob_tx']}%)\n"
                f"🔢 Chẵn/Lẻ: {prediction['chan_le']} ({prediction['prob_cl']}%)\n"
                f"--------------------------------\n"
                f"Kết quả kỳ {ky}:\n"
                f"{' '.join(map(str, dice))} - {tai_xiu}/{chan_le} (Tổng: {total})"
            )
            
            try:
                await bot.send_message(target_group_id, text)
                log(f"Sent prediction for round {ky + 1}")
            except Exception as e:
                log(f"Message send error: {str(e)}")
    except Exception as e:
        log(f"Message handler error: {str(e)}")

# --- Periodic Scanning ---
async def periodic_scan():
    """Periodic full history scan"""
    while True:
        try:
            await asyncio.sleep(3600)  # Scan every hour
            await scan_group()
        except Exception as e:
            log(f"Periodic scan error: {str(e)}")

# --- Main Function ---
async def main():
    max_retries = 3
    retry_count = 0
    
    while retry_count < max_retries:
        try:
            await asyncio.gather(
                client.start(),
                bot.start()
            )
            
            log("Bot started successfully")
            break
            
        except Exception as e:
            retry_count += 1
            log(f"Connection error ({retry_count}/{max_retries}): {str(e)}")
            await asyncio.sleep(5)
    
    if retry_count == max_retries:
        log("Failed to connect after multiple attempts")
        return

    # Initial data load
    data = load_results()
    if not data:
        log("No data found, performing initial scan...")
        await scan_group()
    
    # Start periodic scan task
    asyncio.create_task(periodic_scan())
    
    log("🤖 Bot is now running and monitoring for results...")
    
    await asyncio.gather(
        client.run_until_disconnected(),
        bot.run_until_disconnected()
    )

if __name__ == '__main__':
    try:
        with client, bot:
            client.loop.run_until_complete(main())
    except KeyboardInterrupt:
        log("Bot stopped by user")
    except Exception as e:
        log(f"Fatal error: {str(e)}")
