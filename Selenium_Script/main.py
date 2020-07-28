import csv
import time
import traceback
from random import uniform as rand
from datetime import datetime

from selenium.common.exceptions import TimeoutException
from selenium.webdriver import ActionChains
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait

# region Global variables
global driver
chrome_options = webdriver.ChromeOptions()
chrome_options.add_argument('--no-proxy-server')
driver = webdriver.Chrome(
    'C:/Users/murio/OneDrive/Desktop/webdriver/version81/chromedriver.exe', options=chrome_options)
driver.get("https://www.futbin.com/players?page=1&position=CB,LB,LWB,RB,RWB,CDM,CM,CAM,CF,ST,LM,LW,LF,RM,RW,RF&version=gold_all&sort=Player_Rating&order=desc")  #
wait_time = 1
current_player = {
    "player_id": '',
    "price1_xbox": '',
    "price2_xbox": '',
    "price3_xbox": '',
    "price1_ps4": '',
    "price2_ps4": '',
    "price3_ps4": '',
    "price_time": datetime.now().replace(microsecond=0),
    "price_latest_update": '',
    "price_latest_update_ps4": '',
    "overall_rating": '',
    "name": '',
            "club": '',
            "nation": '',
            "league": '',
            "position": '',
            "skills": '',
            "weakfoot": '',
            "international_rep": '',
            "foot": '',
            "height": '',
            "weight": '',
            "revision": '',
            "defensive_work_rates": '',
            "offensive_work_rates": '',
            "origin": '',
            "birthday": '',

            "pac": '',
            "sho": '',
            "pas": '',
            "dri": '',
            "def": '',
            "phy": '',

            "acceleration": '',
            "sprintspeed": '',
            "positioning": '',
            "finishing": '',
            "shotpower": '',
            "longshots": '',
            "volleys": '',
            "penalties": '',
            "vision": '',
            "crossing": '',
            "freekickaccuracy": '',
            "shortpassing": '',
            "longpassing": '',
            "curve": '',
            "agility": '',
            "balance": '',
            "reactions": '',
            "ballcontrol": '',
            "dribbling": '',
            "composure": '',
            "interceptions": '',
            "headingaccuracy": '',
            "marking": '',
            "standingtackle": '',
            "slidingtackle": '',
            "jumping": '',
            "stamina": '',
            "strength": '',
            "aggression": '',

            "comment": ''
}

current_player_price = {
    "player_id": '',
    "price1": '',
    "price2": '',
    "price3": '',
    "price_time": datetime.now().replace(microsecond=0),
    "price_latest_update": '',
}
export_filepath = 'C:/Users/murio/OneDrive/Desktop/Bachelorthesis/Bachelorthesis/'
export_filename = 'PriceSnapshot_22-04-2020_final.csv'
export_full_path = export_filepath + export_filename
with_header = True

# endregion

# region XPATHS

sort_by_price_xpath = '//*[@id="repTb"]/thead/tr/th[5]/a'
next_page_xpath = '/html/body/div[8]/div/div[4]/div[4]/nav/ul/li[7]/a/span[1]'

# region Attributes
player_id_xpath = '//*[@id="page-data"]'
price1_xpath = '//*[@id="xbox-lowest-1"]'
price2_xpath = '//*[@id="xbox-lowest-2"]'
price3_xpath = '//*[@id="xbox-lowest-3"]'
price_date_xpath = '//*[@id="xbox-updated"]'
overall_rating_xpath = '//*[@id="Player-card"]/div[2]'

pac_xpath = '//*[@id="Player-card"]/div[16]/div[2]'
sho_xpath = '//*[@id="Player-card"]/div[16]/div[5]'
pas_xpath = '//*[@id="Player-card"]/div[16]/div[8]'
dri_xpath = '//*[@id="Player-card"]/div[16]/div[11]'
def_xpath = '//*[@id="Player-card"]/div[16]/div[14]'
phy_xpath = '//*[@id="Player-card"]/div[16]/div[17]'

acceleration_xpath = '//*[@id="sub-acceleration-val-0"]/div[3]'
sprintspeed_xpath = '//*[@id="sub-sprintspeed-val-0"]/div[3]'
positioning_xpath = '//*[@id="sub-positioning-val-0"]/div[3]'
finishing_xpath = '//*[@id="sub-finishing-val-0"]/div[3]'
shotpower_xpath = '//*[@id="sub-shotpower-val-0"]/div[3]'
longshots_xpath = '//*[@id="sub-longshotsaccuracy-val-0"]/div[3]'
volleys_xpath = '//*[@id="sub-volleys-val-0"]/div[3]'
penalties_xpath = '//*[@id="sub-penalties-val-0"]/div[3]'
vision_xpath = '//*[@id="sub-vision-val-0"]/div[3]'
crossing_xpath = '//*[@id="sub-crossing-val-0"]/div[3]'
freekickaccuracy_xpath = '//*[@id="sub-freekickaccuracy-val-0"]/div[3]'
shortpassing_xpath = '//*[@id="sub-shortpassing-val-0"]/div[3]'
longpassing_xpath = '//*[@id="sub-longpassing-val-0"]/div[3]'
curve_xpath = '//*[@id="sub-curve-val-0"]/div[3]'
agility_xpath = '//*[@id="sub-agility-val-0"]/div[3]'
balance_xpath = '//*[@id="sub-balance-val-0"]/div[3]'
reactions_xpath = '//*[@id="sub-reactions-val-0"]/div[3]'
ballcontrol_xpath = '//*[@id="sub-ballcontrol-val-0"]/div[3]'
dribbling_xpath = '//*[@id="sub-dribbling-val-0"]/div[3]'
composure_xpath = '//*[@id="sub-composure-val-0"]/div[3]'
interceptions_xpath = '//*[@id="sub-interceptions-val-0"]/div[3]'
headingaccuracy_xpath = '//*[@id="sub-headingaccuracy-val-0"]/div[3]'
marking_xpath = '//*[@id="sub-marking-val-0"]/div[3]'
standingtackle_xpath = '//*[@id="sub-standingtackle-val-0"]/div[3]'
slidingtackle_xpath = '//*[@id="sub-slidingtackle-val-0"]/div[3]'
jumping_xpath = '//*[@id="sub-jumping-val-0"]/div[3]'
stamina_xpath = '//*[@id="sub-stamina-val-0"]/div[3]'
strength_xpath = '//*[@id="sub-strength-val-0"]/div[3]'
aggression_xpath = '//*[@id="sub-aggression-val-0"]/div[3]'

position_xpath = '//*[@id="Player-card"]/div[4]'
name_xpath = '//*[@id="info_content"]/table/tbody/tr[1]/td'
club_xpath = '//*[@id="info_content"]/table/tbody/tr[2]/td'
nation_xpath = '//*[@id="info_content"]/table/tbody/tr[3]/td'
league_xpath = '//*[@id="info_content"]/table/tbody/tr[4]/td'
skills_xpath = '//*[@id="info_content"]/table/tbody/tr[5]/td'
weakfoot_xpath = '//*[@id="info_content"]/table/tbody/tr[6]/td'
international_rep_xpath = '//*[@id="info_content"]/table/tbody/tr[7]/td'
foot_xpath = '//*[@id="info_content"]/table/tbody/tr[8]/td'
height_xpath = '//*[@id="info_content"]/table/tbody/tr[9]/td'
weight_xpath = '//*[@id="info_content"]/table/tbody/tr[10]/td'
revision_xpath = '//*[@id="info_content"]/table/tbody/tr[11]/td'
defensive_work_rates_xpath = '//*[@id="info_content"]/table/tbody/tr[12]/td'
offensive_work_rates_xpath = '//*[@id="info_content"]/table/tbody/tr[13]/td'
origin_xpath = '//*[@id="info_content"]/table/tbody/tr[15]/td'
birthday_xpath = '//*[@id="info_content"]/table/tbody/tr[17]/td'

origin2_xpath = '//*[@id="info_content"]/table/tbody/tr[14]/td'
birthday2_xpath = '//*[@id="info_content"]/table/tbody/tr[16]/td'

# endregion

# endregion

# region helper functions


def click_by_xpath(xpath, timeout=60):
    wait = WebDriverWait(driver, timeout)
    e = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
    move = ActionChains(driver).move_to_element_with_offset(
        e, rand(1, 5), rand(1, 5))
    move.perform()
    e.click()
    time.sleep(wait_time)


def write_by_xpath(xpath, text, timeout=60):
    wait = WebDriverWait(driver, timeout)
    e = wait.until(EC.element_to_be_clickable((By.XPATH, xpath)))
    move = ActionChains(driver).move_to_element_with_offset(
        e, rand(1, 5), rand(1, 5))
    move.perform()
    e.send_keys(Keys.CONTROL + 'a')
    e.send_keys(Keys.DELETE)
    e.clear()
    e.send_keys(text)


def write_by_id(id_name, text, timeout=60):
    wait = WebDriverWait(driver, timeout)
    e = wait.until(EC.element_to_be_clickable((By.ID, id_name)))
    move = ActionChains(driver).move_to_element_with_offset(
        e, rand(1, 5), rand(1, 5))
    move.perform()
    e.send_keys(Keys.CONTROL + 'a')
    e.send_keys(Keys.DELETE)
    e.clear()
    e.send_keys(text)


def try_write_by_id(id_name, text, timeout=60):
    try:
        write_by_id(id_name, text, timeout)
    except TimeoutException:
        pass


def try_click_by_xpath(xpath, timeout=30):
    try:
        click_by_xpath(xpath, timeout)
    except TimeoutException:
        pass


# endregion

def search_players():
    global current_player
    if with_header:
        with open(export_full_path, 'a',
                  newline='', encoding="utf-8") as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerow(current_player.keys())
    for i in range(1, 160):
        driver.find_element(By.LINK_TEXT, str(i)).send_keys(Keys.RETURN)
        for j in range(30):

            # Access detail page of next player and reset all attributes
            current_player = {
                "player_id": '',
                "price1_xbox": '',
                "price2_xbox": '',
                "price3_xbox": '',
                "price1_ps4": '',
                "price2_ps4": '',
                "price3_ps4": '',
                "price_time": datetime.now().replace(microsecond=0),
                "price_latest_update": '',
                "price_latest_update_ps4": '',
                "overall_rating": '',
                "name": '',
                "club": '',
                "nation": '',
                "league": '',
                "position": '',
                "skills": '',
                "weakfoot": '',
                "international_rep": '',
                "foot": '',
                "height": '',
                "weight": '',
                "revision": '',
                "defensive_work_rates": '',
                "offensive_work_rates": '',
                "origin": '',
                "birthday": '',

                "pac": '',
                "sho": '',
                "pas": '',
                "dri": '',
                "def": '',
                "phy": '',

                "acceleration": '',
                "sprintspeed": '',
                "positioning": '',
                "finishing": '',
                "shotpower": '',
                "longshots": '',
                "volleys": '',
                "penalties": '',
                "vision": '',
                "crossing": '',
                "freekickaccuracy": '',
                "shortpassing": '',
                "longpassing": '',
                "curve": '',
                "agility": '',
                "balance": '',
                "reactions": '',
                "ballcontrol": '',
                "dribbling": '',
                "composure": '',
                "interceptions": '',
                "headingaccuracy": '',
                "marking": '',
                "standingtackle": '',
                "slidingtackle": '',
                "jumping": '',
                "stamina": '',
                "strength": '',
                "aggression": '',

                "comment": ''
            }
            WebDriverWait(driver, 10).until(
                EC.presence_of_element_located((By.CLASS_NAME, 'player_name_players_table')))
            players_web_element = driver.find_elements_by_class_name(
                'player_name_players_table')
            try:
                next_player = players_web_element[j]
                next_player.send_keys(Keys.RETURN)
                try:
                    gather_attributes()
                except Exception as err:
                    current_player["comment"] = traceback.format_exc()
                    export()
                driver.back()
            except IndexError as ie:
                current_player["comment"] = traceback.format_exc()
                export()

# Scrape price data from the player website


def gather_prices():
    if with_header:
        with open(export_full_path, 'a',
                  newline='', encoding="utf-8") as csvFile:
            writer = csv.writer(csvFile, delimiter=';')
            writer.writerow(current_player.keys())
    # get IDs to iterate
    id_list = []
    with open("C:/Users/murio/OneDrive/Desktop/Bachelorthesis/Bachelorthesis/PlayerIDs_split_06122019.csv", encoding="utf-8") as f:
        id_list = [row.split()[0] for row in f]

    for id in id_list:
        driver.get(
            ("https://www.futbin.com/players?page=1&position=CB,LB,LWB,RB,RWB,CDM,CM,CAM,CF,ST,LM,LW,LF,RM,RW,RF&version=gold_all" + id))

        current_player_price["player_id"] = id
        gather_price_data(True)
        export(True)
    # endregion

# Scrape attributes from the player website


def gather_attributes():
    global current_player
    # current_player = {key: '' for key in current_player}
    # region Get Player-ID
    player_id = str(driver.current_url)
    player_id_start = player_id.find("player/") + 7
    player_id = player_id[player_id_start:]
    player_id_end = player_id.find("/")
    player_id = player_id[:player_id_end]
    # endregion

    # region Ratingassignment
    WebDriverWait(driver, 10).until(
        EC.presence_of_element_located((By.XPATH, position_xpath)))
    current_player["position"] = driver.find_element_by_xpath(
        position_xpath).text
    current_player["overall_rating"] = driver.find_element_by_xpath(
        overall_rating_xpath).text
    current_player["pac"] = driver.find_element_by_id("main-pace-val-0").text
    current_player["sho"] = driver.find_element_by_id(
        "main-shooting-val-0").text
    current_player["pas"] = driver.find_element_by_id(
        "main-passing-val-0").text
    current_player["dri"] = driver.find_element_by_id(
        "main-dribblingp-val-0").text
    current_player["def"] = driver.find_element_by_id(
        "main-defending-val-0").text
    current_player["phy"] = driver.find_element_by_id(
        "main-heading-val-0").text
    # endregion

    # region Attributesassignment
    current_player["player_id"] = player_id
    current_player["acceleration"] = driver.find_element_by_id(
        "sub-acceleration-val-0").text
    current_player["sprintspeed"] = driver.find_element_by_id(
        "sub-sprintspeed-val-0").text
    current_player["positioning"] = driver.find_element_by_id(
        "sub-positioning-val-0").text
    current_player["finishing"] = driver.find_element_by_id(
        "sub-finishing-val-0").text
    current_player["shotpower"] = driver.find_element_by_id(
        "sub-shotpower-val-0").text
    current_player["longshots"] = driver.find_element_by_id(
        "sub-longshotsaccuracy-val-0").text
    current_player["volleys"] = driver.find_element_by_id(
        "sub-volleys-val-0").text
    current_player["penalties"] = driver.find_element_by_id(
        "sub-penalties-val-0").text
    current_player["vision"] = driver.find_element_by_id(
        "sub-vision-val-0").text
    current_player["crossing"] = driver.find_element_by_id(
        "sub-crossing-val-0").text
    current_player["freekickaccuracy"] = driver.find_element_by_id(
        "sub-freekickaccuracy-val-0").text
    current_player["shortpassing"] = driver.find_element_by_id(
        "sub-shortpassing-val-0").text
    current_player["longpassing"] = driver.find_element_by_id(
        "sub-longpassing-val-0").text
    current_player["curve"] = driver.find_element_by_id("sub-curve-val-0").text
    current_player["agility"] = driver.find_element_by_id(
        "sub-agility-val-0").text
    current_player["balance"] = driver.find_element_by_id(
        "sub-balance-val-0").text
    current_player["reactions"] = driver.find_element_by_id(
        "sub-reactions-val-0").text
    current_player["ballcontrol"] = driver.find_element_by_id(
        "sub-ballcontrol-val-0").text
    current_player["dribbling"] = driver.find_element_by_id(
        "sub-dribbling-val-0").text
    current_player["composure"] = driver.find_element_by_id(
        "sub-composure-val-0").text
    current_player["interceptions"] = driver.find_element_by_id(
        "sub-interceptions-val-0").text
    current_player["headingaccuracy"] = driver.find_element_by_id(
        "sub-headingaccuracy-val-0").text
    current_player["marking"] = driver.find_element_by_id(
        "sub-marking-val-0").text
    current_player["standingtackle"] = driver.find_element_by_id(
        "sub-standingtackle-val-0").text
    current_player["slidingtackle"] = driver.find_element_by_id(
        "sub-slidingtackle-val-0").text
    current_player["jumping"] = driver.find_element_by_id(
        "sub-jumping-val-0").text
    current_player["stamina"] = driver.find_element_by_id(
        "sub-stamina-val-0").text
    current_player["strength"] = driver.find_element_by_id(
        "sub-strength-val-0").text
    current_player["aggression"] = driver.find_element_by_id(
        "sub-aggression-val-0").text

    # endregion

    # region Infoassignment
    current_player["name"] = driver.find_element_by_xpath(name_xpath).text
    current_player["club"] = driver.find_element_by_xpath(club_xpath).text
    current_player["nation"] = driver.find_element_by_xpath(nation_xpath).text
    current_player["league"] = driver.find_element_by_xpath(league_xpath).text
    current_player["skills"] = driver.find_element_by_xpath(skills_xpath).text
    current_player["weakfoot"] = driver.find_element_by_xpath(
        weakfoot_xpath).text
    temp = driver.find_element_by_xpath(
        '//*[@id="info_content"]/table/tbody/tr[7]/th').text
    if driver.find_element_by_xpath('//*[@id="info_content"]/table/tbody/tr[7]/th').text == "Intl. Rep":
        current_player["international_rep"] = driver.find_element_by_xpath(
            international_rep_xpath).text
        current_player["foot"] = driver.find_element_by_xpath(foot_xpath).text
        current_player["height"] = driver.find_element_by_xpath(
            height_xpath).text
        current_player["weight"] = driver.find_element_by_xpath(
            weight_xpath).text
        current_player["revision"] = driver.find_element_by_xpath(
            revision_xpath).text
        current_player["defensive_work_rates"] = driver.find_element_by_xpath(
            defensive_work_rates_xpath).text
        current_player["offensive_work_rates"] = driver.find_element_by_xpath(
            offensive_work_rates_xpath).text
        current_player["origin"] = driver.find_element_by_xpath(
            origin_xpath).text
        current_player["birthday"] = driver.find_element_by_xpath(
            birthday_xpath).text
    elif driver.find_element_by_xpath('//*[@id="info_content"]/table/tbody/tr[7]/th').text == "Foot":
        current_player["international_rep"] = 'Not available'
        current_player["foot"] = driver.find_element_by_xpath(
            international_rep_xpath).text
        current_player["height"] = driver.find_element_by_xpath(
            foot_xpath).text
        current_player["weight"] = driver.find_element_by_xpath(
            height_xpath).text
        current_player["revision"] = driver.find_element_by_xpath(
            weight_xpath).text
        current_player["defensive_work_rates"] = driver.find_element_by_xpath(
            revision_xpath).text
        current_player["offensive_work_rates"] = driver.find_element_by_xpath(
            defensive_work_rates_xpath).text
        current_player["origin"] = driver.find_element_by_xpath(
            origin2_xpath).text
        current_player["birthday"] = driver.find_element_by_xpath(
            birthday2_xpath).text
    else:
        current_player["comment"] = 'Error in Info-Data'
    # endregion
    gather_price_data()
    export()


def gather_price_data(only_price_data=False):
    # Price editing
    price1 = str(driver.find_element_by_id("xbox-lowest-1").text)
    price2 = str(driver.find_element_by_id("xbox-lowest-2").text)
    price3 = str(driver.find_element_by_id("xbox-lowest-3").text)
    price1_ps4 = str(driver.find_element_by_id("ps-lowest-1").text)
    price2_ps4 = str(driver.find_element_by_id("ps-lowest-2").text)
    price3_ps4 = str(driver.find_element_by_id("ps-lowest-3").text)
    price_latest_update = str(driver.find_element_by_id("xbox-updated").text)
    price_latest_update_ps4 = str(driver.find_element_by_id("ps-updated").text)
    price_latest_update_start = price_latest_update.find(
        'Price Updated: ') + 15
    price_latest_update_start_ps4 = price_latest_update_ps4.find(
        'Price Updated: ') + 15
    price_latest_update = price_latest_update[price_latest_update_start:]
    price_latest_update_ps4 = price_latest_update_ps4[price_latest_update_start_ps4:]

    if only_price_data:
        current_player_price["price1_xbox"] = price1.replace(',', '')
        current_player_price["price2_xbox"] = price2.replace(',', '')
        current_player_price["price3_xbox"] = price3.replace(',', '')

        # Date editing
        current_player_price["price_latest_update_xbox"] = price_latest_update
        current_player_price["price_time"] = datetime.now().replace(
            microsecond=0)
    else:
        current_player["price1_xbox"] = price1.replace(',', '')
        current_player["price2_xbox"] = price2.replace(',', '')
        current_player["price3_xbox"] = price3.replace(',', '')

        # Date editing
        current_player["price_latest_update"] = price_latest_update
        current_player["price_time"] = datetime.now().replace(microsecond=0)

        current_player["price1_ps4"] = price1_ps4.replace(',', '')
        current_player["price2_ps4"] = price2_ps4.replace(',', '')
        current_player["price3_ps4"] = price3_ps4.replace(',', '')

        # Date editing
        current_player["price_latest_update_ps4"] = price_latest_update_ps4


def export(only_price_data=False):
    with open(export_full_path, 'a',
              newline='', encoding="utf-8") as csvFile:
        writer = csv.writer(csvFile, delimiter=';')
        if only_price_data:
            writer.writerow(current_player_price.values())
        else:
            writer.writerow(current_player.values())


# call main function
search_players()
