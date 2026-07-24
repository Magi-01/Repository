from fpdf import FPDF
from datetime import datetime

class PDF(FPDF):
    def header(self):
        self.set_font('Arial', 'B', 12)
        self.cell(0, 10, 'Database Project Report', 0, 1, 'C')
        self.ln(10)

    def chapter_title(self, title):
        self.set_font('Arial', 'B', 14)
        self.cell(0, 10, title, 0, 1, 'L')
        self.ln(5)

    def chapter_body(self, body):
        self.set_font('Arial', '', 12)
        self.multi_cell(0, 10, body)
        self.ln()

    def add_table(self, col_names, data, col_widths=None):
        self.set_font('Arial', 'B', 12)
        if not col_widths:
            col_widths = [self.epw / len(col_names)] * len(col_names)
        line_height = self.font_size * 1.5
        for col_name, col_width in zip(col_names, col_widths):
            self.cell(col_width, line_height, col_name, border=1, align='C')
        self.ln()
        self.set_font('Arial', '', 12)
        for row in data:
            for item, col_width in zip(row, col_widths):
                self.multi_cell(col_width, line_height, item, border=1, align='L', ln=3, max_line_height=line_height)
            self.ln(line_height)
        self.ln()

# Initialize PDF
pdf = PDF()

# Title Page
pdf.add_page()
pdf.set_font("Arial", size=24)
pdf.cell(200, 10, txt="Database Project Report", ln=True, align="C")
pdf.set_font("Arial", size=18)
pdf.cell(200, 10, txt="Author: Your Name", ln=True, align="C")
pdf.cell(200, 10, txt=f"Date: {datetime.today().strftime('%Y-%m-%d')}", ln=True, align="C")

# Section 1: Description of what the database does
pdf.add_page()
pdf.chapter_title("1. Description of what the database does")
description = """
This database manages a game world where players, NPCs (non-player characters), and quests interact. 
Players can join guilds, complete quests, and earn achievements that unlock dimensions. 
NPCs can belong to guilds and initiate quests. The database ensures data integrity and proper relationships 
between entities using triggers and constraints.
"""
pdf.chapter_body(description)

# Section 2: Queries
pdf.chapter_title("2. Queries")
queries_description = """
Triggers:
1. enforce_single_guild_affiliation: Ensures each guild has only one affiliation by checking the count of existing affiliations before inserting a new one.

Queries:
1. ViewNPCGuilds: Retrieves and displays NPCs along with their guild names, defaulting to an empty string if they do not belong to a guild.
"""
pdf.chapter_body(queries_description)

# Section 3: Conceptual schema
pdf.chapter_title("3. Conceptual schema")
conceptual_schema = """
Entities and Attributes:
"""
pdf.chapter_body(conceptual_schema)

conceptual_entities = [
    ["Entity", "Description"],
    ["Player", "Represents players in the game."],
    ["NPC", "Represents non-player characters."],
    ["Guild", "Represents guilds."],
    ["Quest", "Represents quests."],
    ["Achievement", "Represents achievements."],
    ["Dimension", "Represents dimensions."],
    ["Item", "Represents items."],
    ["Player_Item", "Represents items owned by players."]
]

conceptual_relationships = [
    ["Entity (0/1;1/n)", "Relationship", "Entity (0/1;1/n)"],
    ["Player (0/1)", "Complete", "Quest (1/n)"],
    ["Player (0/1)", "Talks", "NPC (1/n)"],
    ["Player (0/1)", "Belong", "Guild (1/n)"],
    ["Player (0/1)", "Own", "Player_Item (1/n)"],
    ["Player (0/1)", "Travel", "Dimension (1/n)"],
    ["Guild (0/1)", "Affiliated", "NPC (1/n)"],
    ["NPC (0/1)", "Initiate", "Quest (1/n)"],
    ["Quest (0/1)", "Check Achievement", "Achievement (1/n)"],
    ["Quest (0/1)", "Check Player_Item", "Player_Item (1/n)"],
    ["Player_Item (0/1)", "Check Achievement", "Achievement (1/n)"],
    ["Player_Item (0/1)", "Refers", "Item (1/n)"],
    ["Achievement (0/1)", "Unlocks", "Dimension (1/n)"]
]

pdf.add_table(conceptual_entities[0], conceptual_entities[1:], col_widths=[40, 100])
pdf.add_table(conceptual_relationships[0], conceptual_relationships[1:], col_widths=[60, 60, 60])

# Section 4: Detailed Redundancy Analysis
pdf.chapter_title("4. Detailed Redundancy Analysis")
redundancy_analysis = """
1. Player_Item and Quest:
    - Redundancy in Checking Items and Achievements:
        - Quest table includes attributes to check player items and achievements.
        - Player_Item also checks achievements, leading to redundancy.
        - Separate tables are used to check relationships between quests, items, and achievements, causing overlapping data.
    - Overlapping Attributes:
        - Quest and Check_Achievement both reference achievement requirements for quests.
        - Player_Item and Check_Achievement both reference player item checks.

2. Guild and NPC:
    - Redundant Relationships:
        - Guild-Affiliated-NPC and Player-Belong-Guild relationships overlap.
        - Information about guild affiliation is duplicated.
"""
pdf.chapter_body(redundancy_analysis)

# Section 5: Removal/addition caused by redundancy analysis
pdf.chapter_title("5. Removal/addition caused by redundancy analysis")
removal_addition = """
- Removal of Overlapping Attributes:
    - Merged the relationships between Quest, Player_Item, and Check_Achievement into a unified structure.
- Simplification of Relationships:
    - Unified guild affiliation information to avoid redundancy.

Resulting Conceptual Schema:

Entities and Attributes:
"""
pdf.chapter_body(removal_addition)

pdf.add_table(conceptual_entities[0], conceptual_entities[1:], col_widths=[40, 100])
pdf.add_table(conceptual_relationships[0], conceptual_relationships[1:], col_widths=[60, 60, 60])

# Section 6: Logical schema
pdf.chapter_title("6. Logical schema")
logical_schema = """
Entities and Attributes:
"""
pdf.chapter_body(logical_schema)

logical_entities = [
    ["Entity", "Attributes"],
    ["Player", "player_id (INT, PK), player_name (VARCHAR), guild_id (INT, FK)"],
    ["NPC", "npc_id (INT, PK), npc_name (VARCHAR), guild_id (INT, FK)"],
    ["Guild", "guild_id (INT, PK), guild_name (VARCHAR)"],
    ["Quest", "quest_id (INT, PK), quest_name (VARCHAR), state (BOOLEAN)"],
    ["Achievement", "achievement_id (INT, PK), achievement_name (VARCHAR)"],
    ["Dimension", "dimension_id (INT, PK), dimension_name (VARCHAR)"],
    ["Item", "item_id (INT, PK), item_name (VARCHAR)"],
    ["Player_Item", "player_item_id (INT, PK), player_id (INT, FK), item_id (INT, FK), state (BOOLEAN)"]
]

logical_relationships = [
    ["Relationship", "Attributes"],
    ["Complete", "player_id (INT, FK), quest_id (INT, FK)"],
    ["Talks", "player_id (INT, FK), npc_id (INT, FK)"],
    ["Belong", "player_id (INT, FK), guild_id (INT, FK)"],
    ["Own", "player_id (INT, FK), player_item_id (INT, FK)"],
    ["Travel", "player_id (INT, FK), dimension_id (INT, FK)"],
    ["Affiliated", "guild_id (INT, FK), npc_id (INT, FK), affiliation (VARCHAR)"],
    ["Initiate", "npc_id (INT, FK), quest_id (INT, FK)"],
    ["Check_Achievement", "quest_id (INT, FK), achievement_id (INT, FK), requires_all_player_items (BOOLEAN)"],
    ["Check_Player_Item", "quest_id (INT, FK), player_item_id (INT, FK)"],
    ["Refers", "player_item_id (INT, FK), item_id (INT, FK)"],
    ["Unlock", "achievement_id (INT, FK), dimension_id (INT, FK)"]
]

pdf.add_table(logical_entities[0], logical_entities[1:], col_widths=[50, 110])
pdf.add_table(logical_relationships[0], logical_relationships[1:], col_widths=[50, 110])

# Section 7: Normalization of logical schema
pdf.chapter_title("7. Normalization of logical schema")
normalization = """
First Normal Form (1NF):
- Player: Each player has a unique player_id, player_name, and guild_id.
- NPC: Each NPC has a unique npc_id, npc_name, and guild_id.
- Guild: Each guild has a unique guild_id and guild_name.
- Quest: Each quest has a unique quest_id, quest_name, and state.
- Achievement: Each achievement has a unique achievement_id and achievement_name.
- Dimension: Each dimension has a unique dimension_id and dimension_name.
- Item: Each item has a unique item_id and item_name.
- Player_Item: Each player item has a unique player_item_id, player_id, item_id, and state.

Second Normal Form (2NF):
- Player: player_name and guild_id are fully dependent on player_id.
- NPC: npc_name and guild_id are fully dependent on npc_id.
- Guild: guild_name is fully dependent on guild_id.
- Quest: quest_name and state are fully dependent on quest_id.
- Achievement: achievement_name is fully dependent on achievement_id.
- Dimension: dimension_name is fully dependent on dimension_id.
- Item: item_name is fully dependent on item_id.
- Player_Item: player_id, item_id, and state are fully dependent on player_item_id.

Third Normal Form (3NF):
- Player: No transitive dependencies exist.
- NPC: No transitive dependencies exist.
- Guild: No transitive dependencies exist.
- Quest: No transitive dependencies exist.
- Achievement: No transitive dependencies exist.
- Dimension: No transitive dependencies exist.
- Item: No transitive dependencies exist.
- Player_Item: No transitive dependencies exist.
"""
pdf.chapter_body(normalization)

# Save the PDF
pdf_file_path = "C:/Users/mutua/Documents/Repository/SQL/Proggetto/sql_database_design.pdf"
pdf.output(pdf_file_path)

pdf_file_path
