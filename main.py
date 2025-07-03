from dotenv import load_dotenv
import discord
from discord.commands import Option
from discord.ext import commands
import os
import random
import math
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple

# keep_alive.py を利用している場合は、このファイルもプロジェクトに含めてください
from keep_alive import keep_alive
keep_alive()

# ---------------- 定数・パラメータ設定 ----------------
load_dotenv()

# 環境変数からトークンを読み込む
BOT_TOKEN = os.getenv("DISCORD_TOKEN") 

RANK_TO_MMR = {
    'アイアン4': 0, 'アイアン3': 100, 'アイアン2': 200, 'アイアン1': 300,
    'ブロンズ4': 400, 'ブロンズ3': 500, 'ブロンズ2': 600, 'ブロンズ1': 700,
    'シルバー4': 800, 'シルバー3': 900, 'シルバー2': 1000, 'シルバー1': 1100,
    'ゴールド4': 1200, 'ゴールド3': 1300, 'ゴールド2': 1400, 'ゴールド1': 1500,
    'プラチナ4': 1600, 'プラチナ3': 1700, 'プラチナ2': 1800, 'プラチナ1': 1900,
    'エメラルド4': 2000, 'エメラルド3': 2100, 'エメラルド2': 2200, 'エメラルド1': 2300,
    'ダイヤモンド4': 2400, 'ダイヤモンド3': 2600, 'ダイヤモンド2': 2800, 'ダイヤモンド1': 3000,
    'マスター': 3200, 'グランドマスター': 3500, 'チャレンジャー': 3800
}
ENTRY_LANES = ["TOP", "JG", "MID", "ADC", "SUP"]

INITIAL_TEMP = 1000.0
COOLING_RATE = 0.998
MAX_ITERATIONS = 50000

W_MMR_STD = 1.0
W_LANE_DIV = 100.0
W_TOP_STD = 500.0

# ---------------- データ構造とグローバル変数 ----------------

@dataclass
class Player:
    id: int
    name: str
    rank: str
    role: str
    mmr: int

entry_list: Dict[int, Player] = {}

# ---------------- チーム分けアルゴリズム (変更なし) ----------------
# (省略... 前回のコードと同じものをここに配置してください)
def calculate_multi_team_score(teams: List[List[Player]], all_players: List[Player]) -> float:
    if not teams: return 0.0
    team_mmr_sums = [sum(p.mmr for p in team) for team in teams]
    e_mmr_std = np.std(team_mmr_sums)
    e_lane_diversity = 0
    for team in teams:
        roles_in_team = {p.role for p in team if p.role in ENTRY_LANES}
        e_lane_diversity += (5 - len(roles_in_team))
    num_teams = len(teams)
    top_player_count = min(len(all_players), num_teams * 2)
    top_players = sorted(all_players, key=lambda p: p.mmr, reverse=True)[:top_player_count]
    top_player_distribution = [len(set(p.id for p in team) & set(p.id for p in top_players)) for team in teams]
    e_top_std = np.std(top_player_distribution)
    score = (W_MMR_STD * e_mmr_std) + (W_LANE_DIV * e_lane_diversity) + (W_TOP_STD * e_top_std)
    return score
# ----------------- チーム分けアルゴリズム（フィルなし・厳格ロール版） -----------------

# ▼▼▼【変更】チーム分けアルゴリズム本体を、フィルしないロジックに全面変更▼▼▼
def balance_multiple_teams(players: List[Player]) -> Tuple[List[List[Player]], List[Player], float]:
    """
    焼きなまし法を用いて、ロールを各1人ずつ揃えたチームを編成する。
    構成に必要なロールの人数が足りない場合、そのプレイヤーは待機メンバーとなる。
    """
    if len(players) < 5:
        return [], players, 0.0

    # --- ステップ1: 編成可能な最大チーム数を決定 ---
    players_by_role: Dict[str, List[Player]] = {role: [] for role in ENTRY_LANES}
    for p in players:
        # エントリー時にロールが保証されている前提
        players_by_role[p.role].append(p)

    # 各ロールの人数をカウント
    role_counts = {role: len(p_list) for role, p_list in players_by_role.items()}
    
    # 最も人数の少ないロールの数が、作れるチーム数の上限
    if not role_counts or min(role_counts.values()) == 0:
        # どのかのロールが0人ならチームは作れない
        num_teams = 0
    else:
        num_teams = min(role_counts.values())

    if num_teams == 0:
        return [], players, 0.0

    # --- ステップ2: チーム分け対象メンバーと待機メンバーを選別 ---
    team_candidates = []
    waiting_players = []

    for role in ENTRY_LANES:
        # 各ロールをMMRの高い順にソート
        sorted_role_players = sorted(players_by_role[role], key=lambda p: p.mmr, reverse=True)
        
        # 上位N人（N=チーム数）をチーム分け対象に
        team_candidates.extend(sorted_role_players[:num_teams])
        # それ以外を待機メンバーに
        waiting_players.extend(sorted_role_players[num_teams:])

    # --- ステップ3: 初期チーム編成 ---
    # team_candidatesを再度ロールごとに分類し、各チームに割り振る
    candidates_by_role: Dict[str, List[Player]] = {role: [] for role in ENTRY_LANES}
    for p in team_candidates:
        candidates_by_role[p.role].append(p)

    current_teams: List[List[Player]] = [[] for _ in range(num_teams)]
    for role in ENTRY_LANES:
        for i in range(num_teams):
            # この時点で必ずプレイヤーはいるはず
            player_to_assign = candidates_by_role[role].pop(0)
            current_teams[i].append(player_to_assign)

    # --- ステップ4: 焼きなまし法によるバランス調整（同ロールスワップ） ---
    best_teams = [list(team) for team in current_teams]
    best_score = calculate_multi_team_score(best_teams, team_candidates)
    temp = INITIAL_TEMP

    for _ in range(MAX_ITERATIONS):
        if temp <= 0.01:
            break

        new_teams = [list(team) for team in current_teams]

        if num_teams >= 2:
            t1_idx, t2_idx = random.sample(range(num_teams), 2)
            role_to_swap = random.choice(ENTRY_LANES)
            p1_idx_in_team, p2_idx_in_team = -1, -1

            for i, p in enumerate(new_teams[t1_idx]):
                if p.role == role_to_swap:
                    p1_idx_in_team = i
                    break
            for i, p in enumerate(new_teams[t2_idx]):
                if p.role == role_to_swap:
                    p2_idx_in_team = i
                    break
            
            # インデックスが見つかった場合のみ交換
            if p1_idx_in_team != -1 and p2_idx_in_team != -1:
                p1 = new_teams[t1_idx][p1_idx_in_team]
                p2 = new_teams[t2_idx][p2_idx_in_team]
                new_teams[t1_idx][p1_idx_in_team], new_teams[t2_idx][p2_idx_in_team] = p2, p1

        new_score = calculate_multi_team_score(new_teams, team_candidates)
        current_score = calculate_multi_team_score(current_teams, team_candidates)
        delta = new_score - current_score

        if delta < 0 or random.random() < math.exp(-delta / temp):
            current_teams = [list(team) for team in new_teams]
            if new_score < best_score:
                best_score = new_score
                best_teams = [list(team) for team in new_teams]

        temp *= COOLING_RATE
        
    return best_teams, waiting_players, best_score

# ---------------- Discordボット本体 ----------------

# ▼▼▼【この3行を追加・変更】▼▼▼
intents = discord.Intents.default()
intents.members = True
intents.message_content = True

bot = discord.Bot(intents=intents)
# ▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲▲

@bot.event
async def on_ready():
    print(f"{bot.user}としてログインしました。")

# --- オートコンプリート用関数 ---
async def rank_autocomplete(ctx: discord.AutocompleteContext):
    all_ranks = list(RANK_TO_MMR.keys())
    user_input = ctx.value.lower()
    return [rank for rank in all_ranks if rank.lower().startswith(user_input)]

# --- コマンド群 ---
@bot.slash_command(name="entry", description="カスタムゲームに参加登録します。")
async def entry(ctx: discord.ApplicationContext, rank: Option(str, "ランクを入力(例:ゴールド)", autocomplete=rank_autocomplete, required=True), role: Option(str, "希望ロールを選択", choices=ENTRY_LANES, required=True)):
    if rank not in RANK_TO_MMR:
        await ctx.respond(f"エラー: '{rank}' は無効なランク名です。候補から選択してください。", ephemeral=True); return
    user = ctx.author
    player = Player(id=user.id, name=user.display_name, rank=rank, role=role, mmr=RANK_TO_MMR[rank])
    is_update = user.id in entry_list
    entry_list[user.id] = player
    action_text = "を更新しました" if is_update else "を受け付けました"
    embed = discord.Embed(title="✅ エントリー完了", description=f"**{user.display_name}** さんのエントリーを{action_text}。", color=discord.Color.blue())
    embed.add_field(name="ランク", value=rank, inline=True); embed.add_field(name="希望ロール", value=role, inline=True)
    embed.set_footer(text=f"現在のエントリー人数: {len(entry_list)}人")
    await ctx.respond(embed=embed, ephemeral=True)

@bot.slash_command(name="withdraw", description="エントリーを取り下げます。")
async def withdraw(ctx: discord.ApplicationContext):
    user_id = ctx.author.id
    if user_id in entry_list:
        del entry_list[user_id]
        await ctx.respond("エントリーを取り下げました。", ephemeral=True)
    else:
        await ctx.respond("あなたはエントリーしていません。", ephemeral=True)

@bot.slash_command(name="status", description="現在のエントリー状況を確認します。")
async def status(ctx: discord.ApplicationContext):
    if not entry_list:
        await ctx.respond("現在、エントリーしている人はいません。", ephemeral=True); return
    embed = discord.Embed(title="エントリー状況", description=f"現在のエントリー人数: **{len(entry_list)}**人", color=discord.Color.gold())
    player_texts = []
    sorted_players = sorted(entry_list.values(), key=lambda p: (ENTRY_LANES.index(p.role), -p.mmr))
    for p in sorted_players: player_texts.append(f"`{p.name:<15}` | {p.rank:<12} | **{p.role}**")
    embed.add_field(name="エントリー一覧", value="\n".join(player_texts) or "なし", inline=False)
    role_counts = {role: 0 for role in ENTRY_LANES}
    for player in entry_list.values(): role_counts[player.role] += 1
    max_count = max(role_counts.values()) if role_counts else 0
    role_status_texts = [f"**{role}**: {count}人 (不足: {max_count - count}人)" for role, count in role_counts.items()]
    embed.add_field(name="ロール状況", value="\n".join(role_status_texts), inline=False)
    await ctx.respond(embed=embed)

class ConfirmView(discord.ui.View):
    def __init__(self, author_id):
        super().__init__(timeout=30); self.value = None; self.author_id = author_id
    async def interaction_check(self, interaction: discord.Interaction) -> bool:
        if interaction.user.id != self.author_id:
            await interaction.response.send_message("コマンドを実行した本人のみ操作できます。", ephemeral=True); return False
        return True
    @discord.ui.button(label="実行", style=discord.ButtonStyle.green)
    async def confirm(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.value = True; self.stop()
        for child in self.children: child.disabled = True
        await interaction.response.edit_message(content="処理を実行します...", view=self)
    @discord.ui.button(label="キャンセル", style=discord.ButtonStyle.grey)
    async def cancel(self, button: discord.ui.Button, interaction: discord.Interaction):
        self.value = False; self.stop()
        for child in self.children: child.disabled = True
        await interaction.response.edit_message(content="キャンセルしました。", view=self)

def create_result_embed(player_list: List[Player], teams: List[List[Player]], remains: List[Player], score: float) -> discord.Embed:
    embed = discord.Embed(title="チーム分け結果", description=f"参加者 {len(player_list)}名から {len(teams)}チームを編成しました。", color=discord.Color.green())
    def format_team(team: List[Player]):
        avg_mmr = sum(p.mmr for p in team) / len(team) if team else 0
        text = f"**平均MMR: {avg_mmr:.0f}**\n"
        sorted_team = sorted(team, key=lambda p: ENTRY_LANES.index(p.role) if p.role in ENTRY_LANES else 99)
        for p in sorted_team: text += f"- `{p.name}` ({p.rank} / {p.role})\n"
        return text
    for i, team in enumerate(teams):
        embed.add_field(name=f"【チーム {chr(65+i)}】", value=format_team(team), inline=True)
    if remains:
        remain_text = "\n".join([f"- `{p.name}` ({p.rank} / {p.role})" for p in remains])
        embed.add_field(name="待機メンバー", value=remain_text, inline=False)
    embed.set_footer(text=f"最適化スコア: {score:.2f} (低いほど良い)")
    return embed

@bot.slash_command(name="divide_teams", description="現在エントリー中のメンバーでチーム分けを行います。")
@commands.has_permissions(administrator=True)
async def divide_teams(ctx: discord.ApplicationContext):
    if len(entry_list) < 5:
        await ctx.respond(f"エラー: 5人未満ではチーム分けできません。(現在: {len(entry_list)}人)", ephemeral=True); return
    view = ConfirmView(ctx.author.id)
    await ctx.respond(f"現在のエントリー {len(entry_list)} 人でチーム分けを実行します。\n**実行後、エントリーリストはクリアされます。**", view=view, ephemeral=True)
    await view.wait()
    if view.value is True:
        player_list = list(entry_list.values())
        entry_list.clear()
        teams, remains, score = balance_multiple_teams(player_list)
        result_embed = create_result_embed(player_list, teams, remains, score)
        await ctx.followup.send(embed=result_embed)
        await ctx.followup.send("エントリーリストをクリアしました。次の募集を開始できます。", ephemeral=True)

@bot.slash_command(name="clear_entries", description="エントリーリストを強制的にリセットします。")
@commands.has_permissions(administrator=True)
async def clear_entries(ctx: discord.ApplicationContext):
    entry_list.clear()
    await ctx.respond("✅ エントリーリストをリセットしました。", ephemeral=True)

@bot.slash_command(name="help", description="BOTのコマンド一覧と使い方を表示します。")
async def help(ctx: discord.ApplicationContext):
    embed = discord.Embed(title=" LoLカスタムチーム分けBOT ヘルプ", description="このBOTで利用できるコマンドの一覧です。", color=discord.Color.og_blurple())
    embed.add_field(name="【👤 参加者向けコマンド】", value=f"`/entry` `rank:<ランク>` `role:<ロール>`\nカスタムに参加登録します。ランク入力時には候補が表示されます。\n\n`/withdraw`\nカスタムへの参加登録を取り下げます。\n\n`/status`\n現在のエントリー状況（参加者一覧、ロール状況）を確認します。\n\n`/help`\nこのヘルプメッセージを表示します。", inline=False)
    embed.add_field(name="【👑 管理者向けコマンド】", value=f"`/divide_teams`\n現在エントリーしているメンバーで、チーム分けを実行します。\n\n`/clear_entries`\nすべてのエントリー情報をリセットします。", inline=False)
    embed.add_field(name="【🧪 テスト用コマンド】", value=f"`/debug` `count:<人数>`\n指定した人数のダミープレイヤーを自動で参加登録させます。", inline=False)
    embed.set_footer(text="困ったときはこのコマンドを実行してください。")
    await ctx.respond(embed=embed)

@bot.slash_command(name="debug", description="指定した人数のダミープレイヤーを自動で参加登録させます。")
@commands.has_permissions(administrator=True)
async def debug(ctx: discord.ApplicationContext, count: Option(int, "登録する人数", required=True, min_value=1, max_value=50)):
    DUMMY_NAMES = ["Ashe","Garen","Lux","Darius","Jinx","Yasuo","Zed","Akali","Teemo","LeeSin","Ahri","Ezreal","Katarina","Riven","Vayne","Thresh","Blitzcrank","Morgana","Yi","Fiora","Irelia","Jax","Malphite","Nasus","Veigar","Annie","Brand","Caitlyn","Jhin","Soraka","Lulu","Nami","Leona","Alistar","Amumu","Chogath","Ekko","Fizz","Graves","Heimerdinger","Kayn","Khazix","Kindred","Lucian","MissFortune","Nocturne","Olaf","Pyke","Quinn","Rengar","Shaco","Sion","Sivir"]
    
    existing_names = {p.name for p in entry_list.values()}
    available_names = [name for name in DUMMY_NAMES if name not in existing_names]
    
    if count > len(available_names):
        # deferの後はfollowupで ephemeral なメッセージを送る
        await ctx.followup.send(f"エラー: 追加できるダミープレイヤーの上限は {len(available_names)} 人です。（名前の重複を避けるため）", ephemeral=True)
        return
        
    names_to_add = random.sample(available_names, count)
    ranks = list(RANK_TO_MMR.keys())
    
    added_players_text = []

    for i in range(count):
        name = names_to_add[i]
        user_id = -random.randint(10000, 99999)
        rank = random.choice(ranks)
        role = random.choice(ENTRY_LANES)
        
        player = Player(id=user_id, name=name, rank=rank, role=role, mmr=RANK_TO_MMR[rank])
        entry_list[user_id] = player
        added_players_text.append(f"`{name}` ({rank} / {role})")

    embed = discord.Embed(
        title=f"🧪 デバッグ: {count}人のダミープレイヤーを登録しました",
        description="\n".join(added_players_text),
        color=discord.Color.orange()
    )
    
    await ctx.response.send_message(embed=embed)
# ボットの実行
if BOT_TOKEN:
    bot.run(BOT_TOKEN)
else:
    print("エラー:環境変数 DISCORD_TOKEN が設定されていません。")