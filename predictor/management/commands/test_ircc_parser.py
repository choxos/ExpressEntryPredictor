from django.core.management.base import BaseCommand
import requests
from bs4 import BeautifulSoup
import re


class Command(BaseCommand):
    help = 'Debug IRCC website parsing to understand table structure'

    def handle(self, *args, **options):
        self.stdout.write('🔍 Debugging IRCC website parsing...')
        self.stdout.write('=' * 60)
        
        url = "https://www.canada.ca/en/immigration-refugees-citizenship/corporate/mandate/policies-operational-instructions-agreements/ministerial-instructions/express-entry-rounds.html"
        
        try:
            # Fetch the page
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            }
            
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            self.stdout.write(f'✅ Fetched page: {len(response.content)} bytes')
            
            # Parse HTML
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Find all tables
            tables = soup.find_all('table')
            self.stdout.write(f'📊 Found {len(tables)} table(s)')
            
            for i, table in enumerate(tables):
                self.stdout.write(f'\n🔍 TABLE {i+1}:')
                
                # Check for headers
                headers_found = []
                header_row = table.find('thead')
                if header_row:
                    for th in header_row.find_all(['th', 'td']):
                        headers_found.append(th.get_text(strip=True))
                else:
                    # Try first row
                    first_row = table.find('tr')
                    if first_row:
                        for cell in first_row.find_all(['th', 'td']):
                            headers_found.append(cell.get_text(strip=True))
                
                self.stdout.write(f'   Headers: {headers_found}')
                
                # Count rows
                all_rows = table.find_all('tr')
                data_rows = all_rows[1:] if len(all_rows) > 1 else all_rows
                self.stdout.write(f'   Total rows: {len(all_rows)}, Data rows: {len(data_rows)}')
                
                # Show first few data rows
                for j, row in enumerate(data_rows[:3]):
                    cells = row.find_all(['td', 'th'])
                    cell_data = [cell.get_text(strip=True) for cell in cells]
                    self.stdout.write(f'   Row {j+1}: {cell_data}')
                
                if len(data_rows) > 3:
                    self.stdout.write(f'   ... and {len(data_rows) - 3} more rows')
            
            # Check for any data-driven content or scripts
            scripts = soup.find_all('script')
            self.stdout.write(f'\n📜 Found {len(scripts)} script tags')
            
            # Look for any mention of draws data in scripts
            for script in scripts:
                if script.string and ('draw' in script.string.lower() or 'invitation' in script.string.lower()):
                    self.stdout.write('🔍 Found potential draws data in JavaScript:')
                    self.stdout.write(script.string[:200] + '...' if len(script.string) > 200 else script.string)
                    break
            
            # Check for any divs that might contain the data
            potential_containers = soup.find_all(['div', 'section'], {'class': re.compile(r'table|data|content', re.I)})
            self.stdout.write(f'\n📦 Found {len(potential_containers)} potential data containers')
            
            for container in potential_containers[:3]:
                if container.get('class'):
                    self.stdout.write(f'   Container classes: {container.get("class")}')
                
        except Exception as e:
            self.stdout.write(f'❌ Error: {e}')
            
        self.stdout.write('\n✅ Debug completed') 