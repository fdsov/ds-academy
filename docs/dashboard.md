# Мой прогресс

Эта страница — твой личный профиль обучения. Весь прогресс хранится **локально в этом браузере** (localStorage): он не уходит на сервер, не синхронизируется между устройствами и виден только тебе. Если открыть курс в другом браузере или очистить данные сайта — счётчики начнутся заново.

<div class="dsa-stats">
<div class="dsa-stat dsa-card"><span class="dsa-stat__val" data-dsa-stat="xp">0</span><span class="dsa-stat__label">Очки опыта (XP)</span></div>
<div class="dsa-stat dsa-card"><span class="dsa-stat__val" data-dsa-stat="streak">0</span><span class="dsa-stat__label">Серия, дней подряд</span></div>
<div class="dsa-stat dsa-card"><span class="dsa-stat__val" data-dsa-stat="done">0</span><span class="dsa-stat__label">Шагов пройдено</span></div>
<div class="dsa-stat dsa-card"><span class="dsa-stat__val" data-dsa-stat="total">0</span><span class="dsa-stat__label">Шагов всего</span></div>
<div class="dsa-stat dsa-card"><span class="dsa-stat__val" data-dsa-stat="pct">0%</span><span class="dsa-stat__label">Общий прогресс</span></div>
</div>

## Прогресс по фазам

Кольцо показывает долю пройденных шагов внутри каждой фазы, практикума и воркшопов.

<div class="dsa-ringgrid">
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="phase0"></span><span class="dsa-ringcell__label">Фаза 0 · Фундамент</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="phase1"></span><span class="dsa-ringcell__label">Фаза 1 · Данные</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="phase2"></span><span class="dsa-ringcell__label">Фаза 2</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="phase3"></span><span class="dsa-ringcell__label">Фаза 3</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="phase4"></span><span class="dsa-ringcell__label">Фаза 4</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="phase5"></span><span class="dsa-ringcell__label">Фаза 5</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="phase6"></span><span class="dsa-ringcell__label">Фаза 6</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="phase7"></span><span class="dsa-ringcell__label">Фаза 7</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="practicum"></span><span class="dsa-ringcell__label">Практикум</span></div>
<div class="dsa-ringcell"><span class="dsa-ring" data-dsa-ring="workshops"></span><span class="dsa-ringcell__label">Воркшопы</span></div>
</div>

## Бейджи

Бейджи открываются автоматически по мере прохождения курса: за первый шаг, серии активных дней и закрытые фазы. Приглушённые — ещё не получены.

<div id="dsa-badges" class="dsa-badges"></div>

## Сброс прогресса

<div class="dsa-reset-block">
<p>Если хочешь начать с чистого листа — можно удалить весь локальный прогресс этого браузера: XP, серию, бейджи, отметки пройденного и состояние флешкарт. Действие необратимо.</p>
<button type="button" id="dsa-reset" class="dsa-btn dsa-btn--ghost">Сбросить прогресс</button>
</div>
