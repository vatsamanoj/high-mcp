import aiosqlite
import asyncio
import json

json_template = """[ 
   { 
     "page": 1, 
     "header": { 
       "Reference No.": "", 
       "": null, 
       "": "" 
     }, 
     "table": { 
       "headers": [ 
         "5. HS Code", 
         "6. Marks and numbers of packages", 
         "7. Number and kind of packages; description of goods", 
         "8. Origin criterion (see notes overleaf)", 
         "9. Gross weight or other quantity", 
         "10. Number and date of invoice", 
         "11. f.o.b. value in US $" 
       ], 
       "rows": [ 
         [ 
         ] 
       ] 
     } 
   } 
 ]"""

async def add_template():
    try:
        # Compact the JSON for the prompt
        compact_json = json.dumps(json.loads(json_template), separators=(',', ':'))
        
        async with aiosqlite.connect("requests.db") as db:
            # Pattern: Matches "Extract invoice data: <content>" or similar
            # Captures the rest of the text as group 1
            pattern = r"(?i)extract (?:invoice|customs) data(?:\s+from)?\s*:\s*(.+)"
            
            # Format: Instructions + Schema + Content placeholder {0} (which is the captured text)
            minimized_format = f"Extract Invoice Data. Output JSON: {compact_json} Input: {{0}}"
            
            description = "Invoice/Customs Data Extraction Template"
            
            await db.execute(
                "INSERT INTO templates (pattern, minimized_prompt_format, description) VALUES (?, ?, ?)",
                (pattern, minimized_format, description)
            )
            await db.commit()
            print(f"Template added successfully.")
            print(f"Pattern: {pattern}")
            print(f"Format Preview: {minimized_format[:100]}...")
            
    except Exception as e:
        print(f"Error adding template: {e}")

if __name__ == "__main__":
    asyncio.run(add_template())
