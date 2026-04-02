async def get_flashscore_matches():
    matches = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=[
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--disable-dev-shm-usage',
                '--disable-gpu',           # ajuda em alguns containers
                '--disable-extensions'
            ]
        )
        page = await browser.new_page()
        try:
            await page.goto("https://www.flashscore.pt/tenis/", timeout=90000)
            await page.wait_for_timeout(12000)  # mais tempo para carregar

            # Tenta clicar na tab "Agendados"
            try:
                tab = await page.query_selector("text=Agendados")
                if tab:
                    await tab.click()
                    await page.wait_for_timeout(8000)
            except:
                pass

            elements = await page.query_selector_all(".event__match")
            for el in elements[:80]:
                try:
                    tour = await el.query_selector(".event__tournament")
                    tournament = (await tour.inner_text()).strip() if tour else "Desconhecido"

                    p1 = await el.query_selector(".event__participant--home")
                    j1 = (await p1.inner_text()).strip() if p1 else "?"

                    p2 = await el.query_selector(".event__participant--away")
                    j2 = (await p2.inner_text()).strip() if p2 else "?"

                    time_el = await el.query_selector(".event__time")
                    horario = (await time_el.inner_text()).strip() if time_el else "?"

                    if horario not in ["AO VIVO", "Terminado", "Cancelado", ""]:
                        superficie = detect_surface(tournament)
                        matches.append({
                            'torneio': tournament,
                            'jogador_1': j1,
                            'jogador_2': j2,
                            'horario': horario,
                            'superficie': superficie
                        })
                except Exception:
                    continue
        finally:
            await browser.close()
    return pd.DataFrame(matches)
