import i18n from 'i18next'
import { initReactI18next } from 'react-i18next'

import es from './locales/es'
import en from './locales/en'
import de from './locales/de'
import fr from './locales/fr'
import ru from './locales/ru'
import zh from './locales/zh'

export const LANGUAGES = [
  { code: 'es', label: 'ES · Español'  },
  { code: 'en', label: 'EN · English'  },
  { code: 'de', label: 'DE · Deutsch'  },
  { code: 'fr', label: 'FR · Français' },
  { code: 'ru', label: 'RU · Русский'  },
  { code: 'zh', label: 'ZH · 中文'     },
] as const

export type LangCode = (typeof LANGUAGES)[number]['code']

i18n
  .use(initReactI18next)
  .init({
    resources: {
      es: { translation: es },
      en: { translation: en },
      de: { translation: de },
      fr: { translation: fr },
      ru: { translation: ru },
      zh: { translation: zh },
    },
    lng:           'es',
    fallbackLng:   'es',
    interpolation: { escapeValue: false },
  })

export default i18n
